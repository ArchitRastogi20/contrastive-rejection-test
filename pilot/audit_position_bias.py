"""Offline audit of option-letter effects in run 3, computed from the committed stage-3 JSONL.

No GPU, no network, no model, no new generation: every number here is a deterministic function
of files already in ``code/results/``. Three questions, each already answerable from fields the
run recorded:

    1. In the unedited condition (R0), does the model's choice land uniformly across candidate
       letters, or does one letter absorb more than its share regardless of content?
    2. The design's primary outcome is whether the model now chooses the edited option. Does that
       outcome hold up within each letter stratum, or is it carried by one slot?
    3. The stage-3 rows carry raw per-token logprobs alongside the derived probabilities. Does the
       contested R2-R4 contrast, and the two content contrasts, look the same in log-odds space as
       in probability space?

    python -m pilot.audit_position_bias

Reuses ``analyze_run3``'s loaders (``load_part``), its discrete-outcome and paired-contrast
plumbing (``discrete_outcomes``, ``discrete_contrast``, ``paired_diffs``), and its margin helper
(``_margin``) rather than reimplementing any of them. scipy is not a dependency of this project
(see ``code/requirements.txt``), so the goodness-of-fit test in section 1 is a seeded permutation
test over the same chi-square statistic, in the style ``audit_probe_missingness.py`` already uses
for its own permutation test.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .analyze_run3 import (
    PARTS,
    _margin,
    discrete_contrast,
    discrete_outcomes,
    load_part,
    paired_diffs,
    short_model,
)
from .config import REPO_ROOT, now_utc, stamp_utc
from .run_experiment import bootstrap_ci_mean_diff

# Same seed the rest of this project's offline audits use, so a reader never has to hunt for it.
SEED = 20260822
RESAMPLES = 5_000
BOOTSTRAP_RESAMPLES = 10_000

CONTENT_CONTRASTS = [
    ("R1", "R2", "content at the rival"),
    ("R3", "R4", "content at the third option"),
]

# A stratum with fewer than this many discordant pairs (McNemar's b+c) is reported but flagged
# as too small to read, rather than interpreted.
TOO_SMALL_DISCORDANT = 5


# ------------------------------------------------------------------------ section 1: R0 letters


def part_letters(items_by_model: dict[str, dict[str, dict[str, dict]]]) -> list[str]:
    """The candidate letters this part's items actually carry, read off R0's own
    ``letter_probe.candidates`` (present whether or not that probe completed -- see the module
    docstring's point about DeepSeek's forced probe never completing in Part B: the candidate
    *labels* are recorded independently of whether the *probabilities* were).

    Verified rather than assumed: every part's rows do in fact carry the same candidate set, but
    this reads it from the data instead of hard-coding "4" or "6".
    """
    letters: set[str] = set()
    for its in items_by_model.values():
        for conds in its.values():
            r0 = conds.get("R0")
            if r0 is None:
                continue
            cands = (r0.get("letter_probe") or {}).get("candidates") or []
            letters.update(cands)
    return sorted(letters)


def tabulate_r0_choices(items: dict[str, dict[str, dict]], letters: list[str]) -> dict[str, int]:
    """R0's ``choice`` letter, tallied over one model's items. Only letters in ``letters`` are
    counted; anything else (an unparseable or out-of-range choice) is silently excluded here and
    counted separately by ``count_excluded_r0_choices`` so the two numbers are never conflated."""
    counts = {letter: 0 for letter in letters}
    for conds in items.values():
        r0 = conds.get("R0")
        if r0 is None:
            continue
        choice = r0.get("choice")
        if choice in counts:
            counts[choice] += 1
    return counts


def count_excluded_r0_choices(items: dict[str, dict[str, dict]], letters: list[str]) -> int:
    """R0 rows whose ``choice`` is missing or not one of this part's candidate letters."""
    excluded = 0
    for conds in items.values():
        r0 = conds.get("R0")
        if r0 is None:
            continue
        if r0.get("choice") not in letters:
            excluded += 1
    return excluded


def chi_square_stat(counts: dict[str, int], letters: list[str]) -> float | None:
    """Pearson chi-square against a uniform expectation over ``letters``."""
    total = sum(counts.get(letter, 0) for letter in letters)
    k = len(letters)
    if total == 0 or k == 0:
        return None
    expected = total / k
    return sum((counts.get(letter, 0) - expected) ** 2 / expected for letter in letters)


def uniform_letter_test(
    counts: dict[str, int], letters: list[str], *, seed: int = SEED, resamples: int = RESAMPLES,
) -> dict:
    """Seeded permutation test of the observed chi-square against a uniform null.

    scipy is not a dependency here (checked against ``code/requirements.txt``), so instead of a
    chi-square-distribution p-value, each resample throws ``total`` independent, uniformly random
    letters and recomputes the same statistic. The p-value is the fraction of resamples whose
    statistic is at least as extreme as the observed one, with the ``+1``/``+1`` correction
    ``audit_probe_missingness.permutation_p_value`` already uses so a p-value of exactly 0 is
    never reported from a finite number of resamples.
    """
    total = sum(counts.get(letter, 0) for letter in letters)
    k = len(letters)
    observed = chi_square_stat(counts, letters)
    if observed is None:
        return {"chi2": None, "p_value": None, "n": total, "resamples": resamples, "seed": seed}
    expected = total / k
    rng = random.Random(seed)
    at_least = 0
    for _ in range(resamples):
        sim = [0] * k
        for _ in range(total):
            sim[rng.randrange(k)] += 1
        stat = sum((c - expected) ** 2 / expected for c in sim)
        if stat >= observed - 1e-12:
            at_least += 1
    return {
        "chi2": observed,
        "p_value": (at_least + 1) / (resamples + 1),
        "n": total,
        "resamples": resamples,
        "seed": seed,
    }


# ---------------------------------------------------------------- section 2: letter stratification


def letter_of_pair(conds: dict[str, dict], a: str, b: str) -> str | None:
    """The shared ``edited_letter`` of conditions ``a`` and ``b`` within one item, or ``None`` if
    either side is missing, unedited, or (unexpectedly) the two sides disagree on which letter was
    edited. R1/R2 always edit the rival's letter and R3/R4 always edit the third option's letter,
    and both are fixed once per item (see the module-level note on shuffling), so a mismatch here
    would itself be an anomaly worth surfacing rather than silently pooling past."""
    ra, rb = conds.get(a), conds.get(b)
    if ra is None or rb is None:
        return None
    la, lb = ra.get("edited_letter"), rb.get("edited_letter")
    if la is None or lb is None or la != lb:
        return None
    return la


def stratify_by_letter(
    pooled_items: dict[str, dict[str, dict]], a: str, b: str,
) -> dict[str, dict[str, dict[str, bool | None]]]:
    """Pooled (across-model) discrete outcomes for contrast ``a``-``b``, split by the letter that
    contrast edited. Pooling models is legitimate here because the option shuffle is a function of
    the item, not the model (see ``data.build_item``): the same item presents the same letter to
    every model."""
    disc = discrete_outcomes(pooled_items)
    strata: dict[str, dict[str, dict[str, bool | None]]] = {}
    for item_id, conds in pooled_items.items():
        letter = letter_of_pair(conds, a, b)
        if letter is None:
            continue
        strata.setdefault(letter, {})[item_id] = disc[item_id]
    return strata


def letter_mismatch_count(pooled_items: dict[str, dict[str, dict]], a: str, b: str) -> int:
    """Items where both sides of the contrast exist but disagree on the edited letter -- expected
    to be zero; reported explicitly rather than assumed."""
    mismatches = 0
    for conds in pooled_items.values():
        ra, rb = conds.get(a), conds.get(b)
        if ra is None or rb is None:
            continue
        la, lb = ra.get("edited_letter"), rb.get("edited_letter")
        if la is not None and lb is not None and la != lb:
            mismatches += 1
    return mismatches


# ------------------------------------------------------------------- section 3: log-odds margins


def margin_outcomes_logodds(items: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """The log-odds variant of ``analyze_run3.margin_outcomes``: the same per-item, per-condition
    change against an R0 baseline, but computed from ``letter_probe.raw_logprobs`` instead of
    ``letter_probe.probs``. Reuses ``analyze_run3._margin`` unchanged -- that helper only needs a
    letter and a ``{letter: float}`` mapping, and a raw-logprob mapping is exactly that shape --
    so this is the same code path fed a different field, not a parallel reimplementation.

    A row whose probe never completed (``letter_probe.complete`` is ``False``) carries
    ``raw_logprobs`` values of ``None`` for every candidate (verified against a real record before
    writing this) and is excluded here on the same ``complete`` flag ``margin_outcomes`` itself
    gates on, so a model whose probe never completes on any row (Part B's DeepSeek-R1-Distill,
    0/425 complete) contributes an empty result here rather than a fabricated one.
    """
    out: dict[str, dict[str, dict]] = {}
    for item_id, conds in items.items():
        r0 = conds.get("R0")
        if r0 is None:
            continue
        r0_probe = r0.get("letter_probe") or {}
        if not r0_probe.get("complete"):
            continue
        r0_logprobs = r0_probe.get("raw_logprobs") or {}
        per_item: dict[str, dict] = {}
        for cond, row in conds.items():
            if cond == "R0":
                continue
            letter = row.get("edited_letter")
            probe = row.get("letter_probe") or {}
            if letter is None or not probe.get("complete"):
                continue
            logprobs = probe.get("raw_logprobs") or {}
            here = _margin(logprobs, letter)
            base = _margin(r0_logprobs, letter)
            if here is None or base is None:
                continue
            per_item[cond] = {"delta_margin_logodds": here - base, "complete": True}
        if per_item:
            out[item_id] = per_item
    return out


# --------------------------------------------------------------------------------- reporting


def _ci(d: dict, places: int = 4) -> str:
    if d.get("ci_low") is None:
        return "n/a"
    return f"{d['mean']:+.{places}f} [{d['ci_low']:+.{places}f}, {d['ci_high']:+.{places}f}]"


def build_report(results: Path) -> str:
    lines: list[str] = []
    w = lines.append

    w("# Option-letter effects in run 3")
    w("")
    w(f"Generated by `python -m pilot.audit_position_bias` at {stamp_utc(now_utc())}, from the")
    w("committed stage-3 JSONL in `code/results/exp3a`, `exp3b`, `exp3c`. No GPU, no network, no")
    w("generation. Every permutation test here uses seed `{}` with `{}` resamples; every".format(
        SEED, RESAMPLES))
    w(f"bootstrap CI uses seed `{SEED}` with `{BOOTSTRAP_RESAMPLES}` resamples, the same discipline")
    w("`analyze_run3.py` uses for its own bootstraps. scipy is not a dependency of this project")
    w("(checked against `code/requirements.txt`), so section 1's goodness-of-fit test is a")
    w("permutation test over the chi-square statistic rather than a library chi-square p-value.")
    w("")

    loaded = {part: load_part(results, part) for part in PARTS}

    # --------------------------------------------------------------------------- section 1
    w("## 1. Option-position bias in the unedited condition (R0)")
    w("")
    w("For each part and model, the distribution of R0's `choice` over candidate letters, against")
    w("the uniform expectation for that part's option count. A model with no position bias should")
    w("show a small chi-square and a large permutation p-value; a model that favours one slot")
    w("regardless of content should not.")
    w("")
    for part in PARTS:
        models = loaded[part]
        letters = part_letters(models)
        w(f"### Part {part} ({len(letters)} options: {', '.join(letters)})")
        w("")
        header = "| model | " + " | ".join(letters) + " | n | excluded | chi2 | permutation p |"
        sep = "|---|" + "---:|" * len(letters) + "---:|---:|---:|---:|"
        w(header)
        w(sep)
        pooled_counts = {letter: 0 for letter in letters}
        pooled_excluded = 0
        for model, items in sorted(models.items()):
            counts = tabulate_r0_choices(items, letters)
            excluded = count_excluded_r0_choices(items, letters)
            for letter in letters:
                pooled_counts[letter] += counts[letter]
            pooled_excluded += excluded
            test = uniform_letter_test(counts, letters)
            cells = " | ".join(str(counts[letter]) for letter in letters)
            chi2_text = "n/a" if test["chi2"] is None else f"{test['chi2']:.3f}"
            p_text = "n/a" if test["p_value"] is None else f"{test['p_value']:.4f}"
            w(f"| {short_model(model)} | {cells} | {test['n']} | {excluded} | {chi2_text} "
              f"| {p_text} |")
        pooled_test = uniform_letter_test(pooled_counts, letters)
        pooled_cells = " | ".join(str(pooled_counts[letter]) for letter in letters)
        w(f"| **pooled** | {pooled_cells} | {pooled_test['n']} | {pooled_excluded} "
          f"| {pooled_test['chi2']:.3f} | {pooled_test['p_value']:.4f} |")
        w("")
    w("`excluded` counts R0 rows whose `choice` was missing or not one of this part's candidate")
    w("letters; none were observed in this run's data, and the column is kept so a future run")
    w("that does have them cannot silently disappear into the letter counts.")
    w("")

    # --------------------------------------------------------------------------- section 2
    w("## 2. Is the content effect a letter artifact?")
    w("")
    w("The design's primary outcome is whether the model now chooses the edited option. This")
    w("breaks that outcome out by which letter the edited option occupied (`edited_letter`), per")
    w("part, for the two content contrasts. Option order is shuffled once per item with a")
    w("per-item seeded RNG (`random.Random(f\"{seed}:{item_id}\")` in `pilot/data.py`'s")
    w("`build_item`) and then held fixed across all five conditions -- `pilot/repair.py` builds")
    w("R0-R4 from that same `Item`'s option list, only appending sentences, never reordering it --")
    w("so letter position varies between items but never within an item. **These strata are")
    w("therefore between-item comparisons**: the letter-D items in a contrast are a different set")
    w("of items from the letter-A items in the same contrast, not the same items seen twice. No")
    w("pooled significance test is computed across strata; the strata are reported so the reader")
    w("can see them directly. Models are pooled within a part because the shuffle is a property of")
    w("the item, not the model -- the same item shows the same letter to every model.")
    w("")
    for part in PARTS:
        models = loaded[part]
        pooled_items = {f"{m}::{i}": conds for m, its in models.items() for i, conds in its.items()}
        for a, b, why in CONTENT_CONTRASTS:
            mismatch = letter_mismatch_count(pooled_items, a, b)
            strata = stratify_by_letter(pooled_items, a, b)
            w(f"### Part {part}, {a}-{b} ({why})")
            w("")
            if mismatch:
                w(f"`{mismatch}` item(s) had disagreeing `edited_letter` between {a} and {b} and")
                w("were excluded from every stratum below.")
                w("")
            w("| letter | n paired | b (favours {a}) | c (favours {b}) | exact p | RD [95% CI] | note |".format(
                a=a, b=b))
            w("|---|---:|---:|---:|---:|---|---|")
            for letter in sorted(strata):
                d = discrete_contrast(strata[letter], a, b)
                too_small = (d["b_a_only"] + d["c_b_only"]) < TOO_SMALL_DISCORDANT
                note = "too small to read" if too_small else ""
                w(f"| {letter} | {d['n_paired']} | {d['b_a_only']} | {d['c_b_only']} "
                  f"| {d['p_exact_two_sided']:.4f} | {_ci(d['rd'], 3)} | {note} |")
            w("")
    w("A stratum flagged `too small to read` has fewer than "
      f"{TOO_SMALL_DISCORDANT} discordant pairs (McNemar's b+c); its point estimate is reported")
    w("above but not interpreted here.")
    w("")

    # --------------------------------------------------------------------------- section 3
    w("## 3. Log-odds margin variant of the contested contrast")
    w("")
    w("The stage-3 rows carry `raw_logprobs` alongside the derived `probs`. This recomputes the")
    w("paired continuous outcome as a log-odds margin -- the target letter's raw logprob minus its")
    w("best competitor's raw logprob, baseline-corrected against R0 exactly the way")
    w("`analyze_run3.margin_outcomes` baseline-corrects the probability-space margin -- for the")
    w("R2-R4 contrast and the two content contrasts, per part, complete-case. This reuses")
    w("`analyze_run3._margin` unchanged, fed `raw_logprobs` instead of `probs`; see")
    w("`margin_outcomes_logodds` in this module.")
    w("")
    w("A row whose forced letter probe never completed carries `raw_logprobs` values of `None`")
    w("for every candidate and is excluded on the same `complete` flag the probability-space")
    w("measure uses. Part B's DeepSeek-R1-Distill-Qwen-14B-AWQ completes on 0 of 425 rows (the")
    w("per-condition completion audit reports the same count) -- this is total")
    w("missingness for that model, not a small-sample gap, and it is reported below as `n = 0` on")
    w("every contrast for that model rather than left as a silent empty cell.")
    w("")
    contrasts = [("R1", "R2", "content at the rival"), ("R3", "R4", "content at the third option"),
                 ("R2", "R4", "the contested contrast")]
    for part in PARTS:
        models = loaded[part]
        pooled_items = {f"{m}::{i}": conds for m, its in models.items() for i, conds in its.items()}
        w(f"### Part {part}")
        w("")
        w("| scope | contrast | n complete-case | mean Δ log-odds margin [95% CI] |")
        w("|---|---|---:|---|")
        scopes = [(short_model(m), its) for m, its in sorted(models.items())]
        scopes.append(("**pooled**", pooled_items))
        for scope, its in scopes:
            marg = margin_outcomes_logodds(its)
            for a, b, _why in contrasts:
                diffs = paired_diffs(marg, a, b, "delta_margin_logodds")
                res = bootstrap_ci_mean_diff(diffs, seed=SEED, n_resamples=BOOTSTRAP_RESAMPLES)
                w(f"| {scope} | {a}-{b} | {res['n']} | {_ci(res)} |")
        w("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=Path("results"), type=Path,
                     help="the results tree holding exp3a, exp3b, exp3c (default: results)")
    ap.add_argument("--out", default=None, type=Path,
                     help="where to write the report (default: POSITION_BIAS_AUDIT.md at the "
                          "tree root)")
    args = ap.parse_args(argv)

    # Report lands at the tree root; this project keeps its write-ups outside the code tree, so
    # the generated file is moved there afterwards rather than written across trees from here.
    out = args.out or (REPO_ROOT / "POSITION_BIAS_AUDIT.md")
    report = build_report(args.results)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
