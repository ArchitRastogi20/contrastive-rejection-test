"""Recompute Part C's discrete matched-pairs results with the corrected letter parser.

Part C's items present six options (A-F). The un-patched `extract.parse_choice` capped its
letter regexes at [A-D], so a bare-letter answer of "E" or "F" was invisible to the letter
path and fell through to the entity-title path -- which recovers a name but not a lone letter.
That left `choice` null on rows where the model in fact named a candidate. This script re-reads
every Part C stage-3 response with the corrected, option-count-aware `parse_choice` (imported
from `extract.py`, not reimplemented) and recomputes every discrete number the study reports for
Part C: the four matched-pairs contrasts, the twelve-test Holm family, and the three
leave-one-model-out checks.

It is read-only against `code/results/`: it loads stage-1 (for the fixed option order) and
stage-3 (for the response text and the original, uncorrected `choice`) and writes nothing back.

Run from the directory that holds `harness/` and `results/` (the `code/` tree in the working
repository, the tree root in the released one):

    python -m harness.reparse_partc

or as a bare script, which puts that directory on `sys.path` itself:

    python harness/reparse_partc.py

Standard library only. No randomness is used, so there is nothing to seed.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

if __package__ in (None, ""):
    # Invoked as a bare script rather than `python -m harness.reparse_partc`: put the `code/`
    # directory (this file's parent's parent) on sys.path so `harness` resolves as a package.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from harness.extract import parse_choice
else:
    from .extract import parse_choice

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(_HERE), "results", "exp3c")

CONTRASTS: list[tuple[str, str]] = [("R1", "R2"), ("R3", "R4"), ("R1", "R3"), ("R2", "R4")]

MODELS = (
    "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
)

# The published Part C numbers (paper, pre-fix), used only as a reproduction check on the
# ORIGINAL `choice` field -- if this script's pairing convention cannot reproduce them, the
# convention differs from what produced the paper and the corrected recompute below would not
# be comparable to it.
PUBLISHED_PARTC: dict[tuple[str, str], tuple[int, int, float]] = {
    ("R1", "R2"): (48, 23, 2.09),
    ("R3", "R4"): (36, 9, 4.00),
    ("R1", "R3"): (59, 36, 1.64),
    ("R2", "R4"): (37, 13, 2.85),
}

# Published exact McNemar p-values for Parts A and B (unaffected by this bug -- see
# reparse_partc's sibling verification for Task 3) -- carried forward unchanged into the
# twelve-test Holm family below.
PART_A_P: dict[tuple[str, str], float] = {
    ("R1", "R2"): 0.004601,
    ("R3", "R4"): 0.001658,
    ("R1", "R3"): 0.056815,
    ("R2", "R4"): 0.013531,
}
PART_B_P: dict[tuple[str, str], float] = {
    ("R1", "R2"): 0.162756,
    ("R3", "R4"): 1.0,
    ("R1", "R3"): 0.164149,
    ("R2", "R4"): 0.855536,
}

# Published leave-one-out exact p for Part C R3-R4, the paper's "only every-drop survivor"
# claim -- used only to state whether that claim still holds after the fix.
PUBLISHED_LOO_C_R3_R4 = {
    "Qwen2.5-7B-Instruct": 0.0118,
    "Llama-3.1-8B-Instruct": 0.0001,  # published as "<0.0001"
    "Mistral-7B-Instruct-v0.3": 0.0070,
}


class _FakeOption:
    """The only thing `parse_choice` reads off an option: its title."""

    __slots__ = ("title",)

    def __init__(self, title: str) -> None:
        self.title = title


class _FakeItem:
    """The minimal stand-in `parse_choice` needs: `.options`, in presentation order."""

    __slots__ = ("options",)

    def __init__(self, titles: list[str]) -> None:
        self.options = [_FakeOption(t) for t in titles]


# --------------------------------------------------------------------------------- loading


def _iter_jsonl(pattern: str):
    for fn in sorted(glob.glob(pattern)):
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def load_option_titles() -> tuple[dict[tuple[str, str], list[str]], list[tuple]]:
    """(item_id, model) -> option_titles, from stage-1 (fixed order, reused by stage-3).

    Verifies the option count implied by `option_titles` is 6 for every Part C item, and that
    a given (item_id, model) never carries two different orders across stage-1 rows (it is
    written once per item and reused unchanged by every condition and every stage).
    """
    titles_by_key: dict[tuple[str, str], list[str]] = {}
    seen: dict[tuple[str, str], list[str]] = {}
    anomalies: list[tuple] = []
    for rec in _iter_jsonl(os.path.join(RESULTS_DIR, "stage1___*.jsonl")):
        key = (rec["item_id"], rec["model"])
        titles = rec["option_titles"]
        if len(titles) != 6:
            anomalies.append((key, len(titles)))
        if key in seen and seen[key] != titles:
            anomalies.append((key, "INCONSISTENT_ORDER"))
        seen[key] = titles
        titles_by_key[key] = titles
    return titles_by_key, anomalies


def load_stage3() -> list[dict]:
    return list(_iter_jsonl(os.path.join(RESULTS_DIR, "stage3___*.jsonl")))


def short_model(full: str) -> str:
    """'/workspace/.hf/models/Qwen2.5-7B-Instruct' -> 'Qwen2.5-7B-Instruct'."""
    return full.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------------- statistics


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p over the n = b + c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    total = sum(math.comb(n, i) for i in range(k + 1))
    return min(2 * total / (2**n), 1.0)


def odds_ratio_ci(b: int, c: int) -> tuple[float, float, float, bool]:
    """OR = b/c with a 95% Wald CI on the log scale; Haldane-Anscombe only if b or c is 0."""
    corrected = b == 0 or c == 0
    bb, cc = (b + 0.5, c + 0.5) if corrected else (b, c)
    orr = bb / cc
    se = math.sqrt(1 / bb + 1 / cc)
    log_or = math.log(orr)
    lo = math.exp(log_or - 1.96 * se)
    hi = math.exp(log_or + 1.96 * se)
    return orr, lo, hi, corrected


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Step-down Holm with monotonicity enforcement over the named family."""
    m = len(pvalues)
    order = sorted(pvalues, key=lambda k: pvalues[k])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(order):  # rank is 0-indexed
        candidate = (m - rank) * pvalues[name]
        running = max(running, candidate)
        adjusted[name] = min(running, 1.0)
    return adjusted


# --------------------------------------------------------------------------------- pairing


def group_by_key(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    """(item_id, model) -> {condition: row}."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        out[(row["item_id"], row["model"])][row["condition"]] = row
    return out


def contrast_counts(
    grouped: dict[tuple[str, str], dict[str, dict]],
    contrast: tuple[str, str],
    choice_field: str,
    model_filter: set[str] | None = None,
) -> tuple[int, int, int]:
    """n (paired, both readable), b, c for one contrast under one choice field.

    b = pairs where condition 1 chose the edited option and condition 2 did not.
    c = the reverse. `choice_field` is 'choice' (original) or 'new_choice' (corrected).
    """
    c1_name, c2_name = contrast
    n = b = c = 0
    for (item_id, model), by_cond in grouped.items():
        if model_filter is not None and short_model(model) not in model_filter:
            continue
        r1 = by_cond.get(c1_name)
        r2 = by_cond.get(c2_name)
        if r1 is None or r2 is None:
            continue
        ch1, ch2 = r1[choice_field], r2[choice_field]
        if ch1 is None or ch2 is None:
            continue
        n += 1
        out1 = ch1 == r1["edited_letter"]
        out2 = ch2 == r2["edited_letter"]
        if out1 and not out2:
            b += 1
        elif out2 and not out1:
            c += 1
    return n, b, c


def report_contrast(n: int, b: int, c: int) -> dict:
    orr, lo, hi, corrected = odds_ratio_ci(b, c)
    p = mcnemar_exact_p(b, c)
    return {
        "n": n, "b": b, "c": c, "OR": orr, "CI_low": lo, "CI_high": hi,
        "haldane_anscombe": corrected, "p": p,
    }


# --------------------------------------------------------------------------------- main


def main() -> None:
    titles_by_key, anomalies = load_option_titles()
    print("=== stage-1 option_titles check (Part C) ===")
    print(f"total (item_id, model) keys: {len(titles_by_key)}")
    print(f"anomalies (option count != 6, or inconsistent order): {len(anomalies)}")
    for a in anomalies[:20]:
        print("  ", a)

    stage3 = load_stage3()
    print(f"\nstage-3 rows loaded: {len(stage3)}")

    missing_titles = 0
    for row in stage3:
        key = (row["item_id"], row["model"])
        titles = titles_by_key.get(key)
        if titles is None:
            missing_titles += 1
            row["new_choice"] = None
            continue
        row["new_choice"] = parse_choice(row["response"], _FakeItem(titles))
    print(f"stage-3 rows with no matching stage-1 option_titles: {missing_titles}")

    orig_null = sum(1 for r in stage3 if r["choice"] is None)
    new_null = sum(1 for r in stage3 if r["new_choice"] is None)
    became_readable = sum(
        1 for r in stage3 if r["choice"] is None and r["new_choice"] is not None
    )
    became_unreadable = sum(
        1 for r in stage3 if r["choice"] is not None and r["new_choice"] is None
    )
    print(f"choice null (original): {orig_null}")
    print(f"choice null (corrected): {new_null}")
    print(f"rows that became readable: {became_readable}")
    print(f"rows that became UNREADABLE (should be 0): {became_unreadable}")

    grouped = group_by_key(stage3)

    # ---- step 4: reproduction check against published Part C numbers, original `choice` ----
    print("\n=== Task 2 step 4: reproduction check (original `choice` field) ===")
    all_reproduced = True
    for contrast in CONTRASTS:
        n, b, c = contrast_counts(grouped, contrast, "choice")
        orr = b / c if c else float("inf")
        pub_b, pub_c, pub_or = PUBLISHED_PARTC[contrast]
        ok = (b, c) == (pub_b, pub_c)
        all_reproduced = all_reproduced and ok
        print(
            f"{contrast[0]}-{contrast[1]}: n={n} b={b} c={c} OR={orr:.4f}  "
            f"published b={pub_b} c={pub_c} OR={pub_or}  MATCH={ok}"
        )
    print(f"ALL FOUR CONTRASTS REPRODUCED: {all_reproduced}")
    if not all_reproduced:
        print(
            "STOP: pairing convention does not reproduce the published Part C numbers from "
            "the ORIGINAL choice field. The corrected recompute below is not valid until this "
            "is resolved."
        )

    # ---- step 5: corrected recompute ----
    print("\n=== Task 2 step 5: corrected Part C contrasts ===")
    corrected_results: dict[tuple[str, str], dict] = {}
    for contrast in CONTRASTS:
        n, b, c = contrast_counts(grouped, contrast, "new_choice")
        res = report_contrast(n, b, c)
        corrected_results[contrast] = res
        print(
            f"{contrast[0]}-{contrast[1]}: n={res['n']} b={res['b']} c={res['c']} "
            f"OR={res['OR']:.4f} CI=[{res['CI_low']:.4f},{res['CI_high']:.4f}] "
            f"HA={res['haldane_anscombe']} p={res['p']:.6g}"
        )

    # ---- also report original-field results (for the before/after table) ----
    print("\n=== original-field Part C contrasts (for before/after table) ===")
    original_results: dict[tuple[str, str], dict] = {}
    for contrast in CONTRASTS:
        n, b, c = contrast_counts(grouped, contrast, "choice")
        res = report_contrast(n, b, c)
        original_results[contrast] = res
        print(
            f"{contrast[0]}-{contrast[1]}: n={res['n']} b={res['b']} c={res['c']} "
            f"OR={res['OR']:.4f} CI=[{res['CI_low']:.4f},{res['CI_high']:.4f}] "
            f"HA={res['haldane_anscombe']} p={res['p']:.6g}"
        )

    # ---- step 6: twelve-test Holm family ----
    print("\n=== Task 2 step 6: twelve-test Holm family ===")
    family: dict[str, float] = {}
    for contrast, p in PART_A_P.items():
        family[f"A {contrast[0]}-{contrast[1]}"] = p
    for contrast, p in PART_B_P.items():
        family[f"B {contrast[0]}-{contrast[1]}"] = p
    for contrast, res in corrected_results.items():
        family[f"C {contrast[0]}-{contrast[1]}"] = res["p"]
    adjusted = holm(family)
    for name in sorted(family, key=lambda k: family[k]):
        print(f"{name}: raw p={family[name]:.6g}  Holm-adjusted={adjusted[name]:.6g}")

    # ---- step 7: leave-one-model-out, corrected Part C ----
    print("\n=== Task 2 step 7: leave-one-model-out (corrected Part C) ===")
    loo: dict[str, dict[tuple[str, str], dict]] = {}
    for dropped in MODELS:
        kept = {m for m in MODELS if m != dropped}
        loo[dropped] = {}
        print(f"-- dropping {dropped} --")
        for contrast in CONTRASTS:
            n, b, c = contrast_counts(grouped, contrast, "new_choice", model_filter=kept)
            res = report_contrast(n, b, c)
            loo[dropped][contrast] = res
            print(
                f"  {contrast[0]}-{contrast[1]}: n={res['n']} b={res['b']} c={res['c']} "
                f"OR={res['OR']:.4f} p={res['p']:.6g}"
            )

    print("\n=== C R3-R4 every-drop-survivor check ===")
    r34_ps = {dropped: loo[dropped][("R3", "R4")]["p"] for dropped in MODELS}
    survives_every_drop = all(p < 0.05 for p in r34_ps.values())
    for dropped, p in r34_ps.items():
        pub = PUBLISHED_LOO_C_R3_R4[dropped]
        print(f"  drop {dropped}: corrected p={p:.6g}   published p={pub}")
    print(f"C R3-R4 significant under every single-model drop (p<0.05): {survives_every_drop}")

    # also check whether any OTHER corrected Part C contrast is significant under every drop,
    # since the paper's claim is that R3-R4 is the ONLY one
    print("\n=== every-drop significance, all four corrected Part C contrasts ===")
    for contrast in CONTRASTS:
        ps = [loo[dropped][contrast]["p"] for dropped in MODELS]
        every = all(p < 0.05 for p in ps)
        print(f"  {contrast[0]}-{contrast[1]}: drop-p={[f'{p:.4g}' for p in ps]} every_drop_sig={every}")


if __name__ == "__main__":
    main()
