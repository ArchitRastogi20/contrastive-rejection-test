"""Does a switch after a date repair land on the option the inserted date makes correct?

Most built items pose a date or order comparison ("Who died first, X or Y?"), and about half
of those receive a date repair. The inserted date is borrowed from a sibling profile and need
not be true, so on such an item it can change which option is actually correct under the
question, and a model that switches to the repaired option may simply be using the new
evidence rather than responding to its own stated reason. This module measures that
directly, with two deterministic checks and no judge:

1. **Stratified contrasts** (committed JSONL only). Each content contrast (R1-R2, R3-R4) is
   recomputed inside and outside the stratum "order question with a date repair". If the
   content effect lived only inside that stratum, question relevance would explain it.

2. **Correctness direction** (needs ``--data-file``). For each built item in the stratum whose
   question directly compares two of the options on the repaired attribute, the inserted
   year is compared against the other named option's year in the direction the question
   asks, giving a verdict: the edited option *becomes correct*, *stays incorrect*, or the
   case is *undeterminable*. That verdict is crossed with whether the model switched to the
   edited option, per condition (R1 at the rival, R3 at the third option), with Fisher's
   exact test on the resulting 2x2.

    python -m harness.audit_date_correctness                       # stratified contrasts only
    python -m harness.audit_date_correctness --data-file dump.jsonl  # adds the direction check

Every classification rule below is a regular expression printed in the report next to a
sample, so it can be audited rather than trusted. No GPU, no network.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from . import config as C
from . import extract
from .analyze_run3 import PARTS, discrete_outcomes, load_part, mcnemar, odds_ratio
from .audit_instrument_defects import _reconstruct_part
from .question_relevance import DATE_ATTRIBUTES, ORDER_QUESTION, question_by_item

CONTENT_CONTRASTS = (("R1", "R2"), ("R3", "R4"))

# Which way the comparison points. Whole words, so "latest" does not also match "later".
EARLIER_WINS = re.compile(r"\b(first|earlier|earliest|before)\b", re.IGNORECASE)
LATER_WINS = re.compile(r"\b(later|latest|last|after|more recently)\b", re.IGNORECASE)

# Which attribute the question compares. A question about a *related* entity's date (the
# director's death, the father's birth) is not answerable by a date inserted into the option's
# own profile, so those are classed separately rather than scored.
QUESTION_ATTRIBUTE = (
    ("date_of_birth", re.compile(r"\b(born|birth)\b", re.IGNORECASE)),
    ("date_of_death", re.compile(r"\b(died|death)\b", re.IGNORECASE)),
)
RELATED_ENTITY = re.compile(
    r"\b(director|producer|composer|performer|screenwriter|editor|father|mother|spouse|"
    r"husband|wife|child|son|daughter|sibling|brother|sister|founder)\b",
    re.IGNORECASE,
)

_YEAR = re.compile(r"(?<![0-9])(1[0-9]{3}|20[0-9]{2})(?![0-9])")
# "X (12 May 1901 - 3 June 1970) was ..." -- the corpus's commonest way of stating both dates.
_LIFESPAN = re.compile(
    r"\(([^()]*?(?<![0-9])(1[0-9]{3}|20[0-9]{2})(?![0-9])[^()]*?)[-–—]"
    r"([^()]*?(?<![0-9])(1[0-9]{3}|20[0-9]{2})(?![0-9])[^()]*?)\)"
)
_BORN_YEAR = re.compile(r"\b(?:born|b\.)\b[^.;]*?(?<![0-9])(1[0-9]{3}|20[0-9]{2})(?![0-9])", re.IGNORECASE)
_DIED_YEAR = re.compile(r"\b(?:died|d\.|death)\b[^.;]*?(?<![0-9])(1[0-9]{3}|20[0-9]{2})(?![0-9])", re.IGNORECASE)


# ------------------------------------------------------------------ the deterministic rules


def norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()


def question_direction(question: str) -> str | None:
    """'earlier', 'later', or None when the question points both ways or neither."""
    e, l = bool(EARLIER_WINS.search(question)), bool(LATER_WINS.search(question))
    if e == l:
        return None
    return "earlier" if e else "later"


def question_attribute(question: str) -> str | None:
    hits = [attr for attr, rx in QUESTION_ATTRIBUTE if rx.search(question)]
    return hits[0] if len(hits) == 1 else None


def options_named_in(question: str, option_titles: list[str]) -> list[int]:
    """Indices of the options whose full title appears in the question, after stripping a
    trailing parenthetical disambiguator, which the corpus's questions never carry."""
    q = norm_title(question)
    out = []
    for i, t in enumerate(option_titles):
        base = norm_title(re.sub(r"\s*\([^)]*\)\s*$", "", t))
        if base and re.search(r"(?<![a-z0-9])" + re.escape(base) + r"(?![a-z0-9])", q):
            out.append(i)
    return out


def year_in_sentence(sentence: str) -> int | None:
    m = _YEAR.search(sentence)
    return int(m.group(1)) if m else None


def year_in_profile(profile: str, attribute: str) -> int | None:
    """The option's own year for `attribute`, read from its unedited profile. Cue-anchored first
    ("born ... 1901", "died ... 1970"), then the parenthetical lifespan, which states birth
    first and death second. None when neither rule fires."""
    rx = _BORN_YEAR if attribute == "date_of_birth" else _DIED_YEAR
    m = rx.search(profile)
    if m:
        return int(m.group(1))
    m = _LIFESPAN.search(profile)
    if m:
        return int(m.group(2) if attribute == "date_of_birth" else m.group(4))
    return None


def verdict(direction: str, inserted_year: int, other_year: int) -> str:
    if inserted_year == other_year:
        return "undeterminable: tie"
    wins = inserted_year < other_year if direction == "earlier" else inserted_year > other_year
    return "becomes correct" if wins else "stays incorrect"


def classify(entry: dict, condition: str, attribute: str) -> dict:
    """One reconstructed built item, one edited condition (R1 or R3) -> a verdict and why."""
    q = entry["question"]
    titles = entry["option_titles"]
    edited_idx = entry.get(f"{condition}_edited_idx")
    sentence = entry.get(f"{condition}_sentence", "")
    out = {"condition": condition, "edited_idx": edited_idx}
    if not ORDER_QUESTION.search(q) or attribute not in DATE_ATTRIBUTES:
        out["class"] = "outside stratum"
        return out
    if RELATED_ENTITY.search(q):
        out["class"] = "question compares a related entity's date"
        return out
    q_attr = question_attribute(q)
    if q_attr is None:
        out["class"] = "question attribute unreadable"
        return out
    if q_attr != attribute:
        out["class"] = "question compares a different attribute"
        return out
    named = options_named_in(q, titles)
    if len(named) != 2:
        out["class"] = f"question names {len(named)} options, not 2"
        return out
    if edited_idx not in named:
        out["class"] = "edited option not named in question"
        return out
    direction = question_direction(q)
    if direction is None:
        out["class"] = "direction unreadable"
        return out
    inserted = year_in_sentence(sentence)
    if inserted is None:
        out["class"] = "no year in inserted sentence"
        return out
    other_idx = named[0] if named[1] == edited_idx else named[1]
    other = year_in_profile(entry["profiles"][other_idx], attribute)
    if other is None:
        out["class"] = "other option's year not extractable"
        return out
    out.update({"class": "scored", "direction": direction, "inserted_year": inserted,
                "other_idx": other_idx, "other_year": other,
                "verdict": verdict(direction, inserted, other)})
    return out


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]], summing every table at least as
    improbable as the observed one under the hypergeometric null. Stdlib only."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)

    p_obs = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p for p in (prob(x) for x in range(lo, hi + 1)) if p <= p_obs + 1e-12))


# ----------------------------------------------------------------- check 1: stratification


def in_stratum(question: str, attribute: str | None) -> bool:
    return bool(ORDER_QUESTION.search(question)) and attribute in DATE_ATTRIBUTES


def stratified_contrasts(results: Path) -> dict:
    """Pooled b, c, OR and exact p for each content contrast, inside and outside the
    order-question-with-date-repair stratum, per part. Choices are re-derived by
    ``load_part`` from raw text, so this is the post-parser-fix reading."""
    questions = question_by_item(results)
    report: dict = {}
    for part in PARTS:
        loaded = load_part(results, part)
        strata = {"inside": {}, "outside": {}}
        for model, items in loaded.items():
            outcomes = discrete_outcomes(items)
            for item_id, conds in items.items():
                attribute = next((r.get("attribute") for r in conds.values() if r.get("attribute")), None)
                q = questions.get((item_id, model), "")
                key = "inside" if in_stratum(q, attribute) else "outside"
                strata[key][f"{model}::{item_id}"] = outcomes[item_id]
        report[part] = {}
        for a, b in CONTENT_CONTRASTS:
            report[part][f"{a}-{b}"] = {}
            for key, outcomes in strata.items():
                mc = mcnemar(outcomes, a, b)
                orr = odds_ratio(mc["b_a_only"], mc["c_b_only"])
                report[part][f"{a}-{b}"][key] = {
                    "n_items": len(outcomes), "b": mc["b_a_only"], "c": mc["c_b_only"],
                    "or": orr["or"], "or_lo": orr["lo"], "or_hi": orr["hi"],
                    "p_exact": mc["p_exact_two_sided"],
                }
    return report


# ------------------------------------------------------------ check 2: correctness direction


def reconstruct_all(results: Path, data_file: Path) -> list[dict]:
    # Same pools `audit_instrument_defects.build_report` uses; see its comment for why Part A's
    # lookup pool is larger than its corpus index.
    out: list[dict] = []
    for part, corpus_n, n_options, lookup_n in (("A", 400, 4, 1800), ("B", 1200, 4, 1200), ("C", 1800, 6, 1800)):
        out.extend(_reconstruct_part(results, data_file, part, corpus_n, n_options, lookup_n))
    return out


def direction_check(results: Path, reconstructed: list[dict]) -> dict:
    """Cross each scored item's verdict with whether the model switched to the edited option."""
    switched: dict[tuple[str, str, str], dict[str, bool | None]] = {}
    attribute_of: dict[tuple[str, str, str], str | None] = {}
    for part in PARTS:
        for model, items in load_part(results, part).items():
            outcomes = discrete_outcomes(items)
            for item_id, conds in items.items():
                switched[(part, model, item_id)] = outcomes[item_id]
                attribute_of[(part, model, item_id)] = next(
                    (r.get("attribute") for r in conds.values() if r.get("attribute")), None)

    # classes[cond][class] = [items, of which switched to the edited option]
    classes: dict[str, dict[str, list[int]]] = {"R1": {}, "R3": {}}
    tables: dict[str, dict[str, dict[str, list[int]]]] = {}
    scored_rows: list[dict] = []
    for entry in reconstructed:
        if not entry.get("reconstructed"):
            continue
        key = (entry["part"], entry["model"], entry["item_id"])
        attribute = attribute_of.get(key)
        for cond in ("R1", "R3"):
            c = classify(entry, cond, attribute)
            flip = switched.get(key, {}).get(cond)
            cell = classes[cond].setdefault(c["class"], [0, 0])
            cell[0] += 1
            cell[1] += int(bool(flip))
            if c["class"] != "scored":
                continue
            if flip is None:
                classes[cond].setdefault("scored but choice unreadable", [0, 0])[0] += 1
                continue
            for scope in ("pooled", entry["part"]):
                cell = tables.setdefault(cond, {}).setdefault(scope, {})
                cell.setdefault(c["verdict"], [0, 0])[0 if flip else 1] += 1
            scored_rows.append({**c, "part": entry["part"], "model": entry["model"].split("/")[-1],
                                "item_id": entry["item_id"], "question": entry["question"],
                                "sentence": entry.get(f"{cond}_sentence", ""), "switched": flip})

    summary: dict = {}
    for cond, scopes in tables.items():
        summary[cond] = {}
        for scope, cells in scopes.items():
            a, b = cells.get("becomes correct", [0, 0])
            c_, d = cells.get("stays incorrect", [0, 0])
            summary[cond][scope] = {
                "becomes_correct_switched": a, "becomes_correct_not": b,
                "stays_incorrect_switched": c_, "stays_incorrect_not": d,
                "switch_rate_if_correct": a / (a + b) if a + b else None,
                "switch_rate_if_incorrect": c_ / (c_ + d) if c_ + d else None,
                "switches_toward_newly_correct_share": a / (a + c_) if a + c_ else None,
                "fisher_p": fisher_exact(a, b, c_, d),
                "ties": sum(v[0] + v[1] for k, v in cells.items() if k.startswith("undeterminable")),
            }
    return {"classes": classes, "summary": summary, "scored_rows": scored_rows,
            "attribute_of": attribute_of}


# ------------------------------------------------ check 3: the date cue inside a longer word


def cue_only_embedded(sentence: str, attribute: str) -> bool | None:
    """True when the sentence carries a cue for `attribute` only as a substring of a longer
    word ("died" in "studied", "born" in "Osborne"), which is exactly what
    `extract.attribute_in_profile`'s unbounded `cue in text` test accepts. False when a
    whole-word cue is present. None when no cue is present at all."""
    low = f" {sentence.casefold()} "
    cues = [c for c in extract.ATTRIBUTES[attribute] if len(c) > 2]
    if any(re.search(r"(?<![a-z])" + re.escape(c) + r"(?![a-z])", low) for c in cues):
        return False
    if any(c in low for c in cues):
        return True
    return None


def embedded_cue_check(reconstructed: list[dict], attribute_of: dict) -> dict:
    out: dict[str, dict[str, int]] = {}
    examples: list[str] = []
    for entry in reconstructed:
        if not entry.get("reconstructed"):
            continue
        attribute = attribute_of.get((entry["part"], entry["model"], entry["item_id"]))
        if attribute not in DATE_ATTRIBUTES:
            continue
        for cond in ("R1", "R3"):
            sentence = entry.get(f"{cond}_sentence", "")
            if not sentence:
                continue
            verdict_ = cue_only_embedded(sentence, attribute)
            label = {False: "whole-word cue", True: "cue only inside a longer word", None: "no cue"}[verdict_]
            cell = out.setdefault(cond, {})
            cell[label] = cell.get(label, 0) + 1
            if verdict_ and len(examples) < 8:
                examples.append(f"[{attribute}, {cond}] {sentence[:140]}")
    return {"counts": out, "examples": examples}


# ----------------------------------------------------------------------------- the report


def _fmt(x: float | None, places: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def build_report(results: Path, data_file: Path | None) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {"seed": C.SEED}

    w("# Date repairs and question correctness")
    w("")
    w("Generated by `python -m harness.audit_date_correctness`. No GPU, no network.")
    w(f"Order-question rule: `{ORDER_QUESTION.pattern}`; date attributes: {sorted(DATE_ATTRIBUTES)}.")
    w("")
    w("## 1. Content contrasts inside and outside the date-repair stratum")
    w("")
    w("Stratum = question matches the order rule AND the repaired attribute is a date. Pooled")
    w("across models; choices re-derived from raw text (post parser fixes).")
    w("")
    strat = stratified_contrasts(results)
    payload["stratified"] = strat
    w("| Part | Contrast | stratum | items | b | c | OR [95% CI] | exact p |")
    w("|---|---|---|---:|---:|---:|---|---:|")
    for part, contrasts in strat.items():
        for name, cells in contrasts.items():
            for key, r in cells.items():
                w(f"| {part} | {name} | {key} | {r['n_items']} | {r['b']} | {r['c']} | "
                  f"{r['or']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] | {r['p_exact']:.4f} |")
    w("")

    w("## 2. Does the switch land on the option the inserted date makes correct?")
    w("")
    if data_file is None:
        w("Not computed: pass `--data-file` to rebuild the inserted sentences.")
        return "\n".join(lines), payload

    reconstructed = reconstruct_all(results, data_file)
    n_ok = sum(1 for r in reconstructed if r.get("reconstructed"))
    w(f"Reconstructed {n_ok}/{len(reconstructed)} built items.")
    w("")
    w("Rules, in order: the question must not compare a related entity's date "
      f"(`{RELATED_ENTITY.pattern}`); its attribute (`born|birth` / `died|death`) must equal the")
    w("repaired attribute; exactly two option titles must appear in it, one of them the edited")
    w(f"option; its direction is read by `{EARLIER_WINS.pattern}` against `{LATER_WINS.pattern}`;")
    w("the inserted year is the first year in the inserted sentence; the other named option's")
    w("year is read from its unedited profile, cue-anchored first, then from a parenthetical")
    w("lifespan. Anything the rules cannot read is counted, not guessed.")
    w("")
    d = direction_check(results, reconstructed)
    payload["direction"] = {"classes": d["classes"], "summary": d["summary"]}
    payload["embedded_cue"] = embedded_cue_check(reconstructed, d["attribute_of"])
    payload["scored_rows"] = d["scored_rows"]
    for cond in ("R1", "R3"):
        label = "the named rival" if cond == "R1" else "the third option"
        w(f"### {cond} (edit at {label}): classification of reconstructed built items")
        w("")
        for k, v in sorted(d["classes"][cond].items(), key=lambda kv: -kv[1][0]):
            w(f"- {k}: {v[0]} items, {v[1]} switched to the edited option")
        w("")
        for scope, s in sorted(d["summary"].get(cond, {}).items(), key=lambda kv: (kv[0] != "pooled", kv[0])):
            w(f"**{cond}, {scope}.** becomes correct: {s['becomes_correct_switched']} switched / "
              f"{s['becomes_correct_not']} not (rate {_fmt(s['switch_rate_if_correct'])}); "
              f"stays incorrect: {s['stays_incorrect_switched']} switched / {s['stays_incorrect_not']} not "
              f"(rate {_fmt(s['switch_rate_if_incorrect'])}); share of switches toward the newly correct "
              f"option {_fmt(s['switches_toward_newly_correct_share'])}; Fisher p = {s['fisher_p']:.4f}; "
              f"ties {s['ties']}.")
            w("")
    w("### The date cue inside a longer word")
    w("")
    w("`extract.attribute_in_profile` tests `cue in text` with no word boundary, so a sentence")
    w("counts as stating a death date when it contains \"studied\" and a year. Date-repair sentences")
    w("that carry the cue only this way, by condition:")
    w("")
    for cond, cells in payload["embedded_cue"]["counts"].items():
        w(f"- {cond}: " + ", ".join(f"{k} {v}" for k, v in sorted(cells.items())))
    w("")
    for ex in payload["embedded_cue"]["examples"]:
        w(f"- {ex}")
    w("")
    w("### Sample of scored rows, so the rules can be audited")
    w("")
    for r in d["scored_rows"][:12]:
        w(f"- [{r['part']} {r['condition']} {r['model']}] {r['question']}  ->  inserted {r['inserted_year']}, "
          f"other {r['other_year']}, {r['direction']} wins: **{r['verdict']}**, switched={r['switched']}")
        w(f"  inserted: \"{r['sentence'][:160]}\"")
    return "\n".join(lines), payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=C.CODE_ROOT / "results")
    ap.add_argument("--data-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the Markdown report here")
    ap.add_argument("--json", type=Path, default=None, help="write the payload here")
    args = ap.parse_args(argv)
    text, payload = build_report(args.results, args.data_file)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
