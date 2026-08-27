"""Whether the content contrasts (R1-R2, R3-R4) measure relevance or a relation mismatch.

No GPU, no network, no generation: every number here is a deterministic function of the
committed stage-3 JSONL in ``code/results/exp3a`` and ``code/results/exp3c``.

Each edited condition inserts one sentence into a candidate's profile. R1 and R3 insert a
sentence carrying the attribute the model itself named as missing; R2 and R4 insert a
length-matched sentence that is irrelevant to that item -- and that irrelevant sentence states
a parentage relation (child / father / mother) in the large majority of items, regardless of
what attribute the model actually named. Since the named attribute is drawn from a wider set
that includes parentage only rarely, "relevant vs. irrelevant" is, in most items, simultaneously
"named relation vs. parentage relation": a template difference the R1-R2 / R3-R4 contrasts could
be picking up instead of, or in addition to, relevance itself.

This splits items by whether the model's own named attribute is itself a parentage relation.
Where it is, the irrelevant control is relation-matched to the repair sentence, and any surviving
effect is attributable to relevance rather than to template mismatch. Where it is not, the two
remain confounded. Comparing the contrasts across the two strata is the check.

    python -m pilot.analyze_relation_stratum                  # both parts, full report
    python -m pilot.analyze_relation_stratum --part a         # Part A only

Discrete measure: McNemar b/c on `chosen_is_edited`, matched-pairs odds ratio (Haldane-Anscombe
+0.5 on both cells only when a cell is zero), Wald CI on the log odds ratio, exact two-sided
binomial p. Continuous measure: paired mean difference in `delta_p_edited`, complete-case (both
conditions non-null and both rows' `letter_probe.complete` true), percentile bootstrap CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_run3 import odds_ratio, short_model
from .config import REPO_ROOT
from .run_experiment import bootstrap_ci_mean_diff, mcnemar

SEED = 20260822
RESAMPLES = 10000

PARTS = {
    "a": ("exp3a", "Part A, 4 options"),
    "c": ("exp3c", "Part C, 6 options"),
}
CONTRASTS = [("R1", "R2"), ("R3", "R4")]
STRATA = ["parentage", "non_parentage", "pooled"]
STRATUM_LABEL = {
    "parentage": "named attribute is parentage (child/father/mother)",
    "non_parentage": "named attribute is not parentage",
    "pooled": "both strata combined (check)",
}

# The named attribute is drawn from {date of birth, date of death, parentage, director credit}
# per the design; parentage is realised in the data as exactly these three relation fields.
PARENTAGE_ATTRIBUTES = {"child", "father", "mother"}


# --------------------------------------------------------------------------------- loading


def _rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_part(results: Path, directory: str) -> dict[str, dict[str, dict]]:
    """``{"model::item_id": {condition: row}}`` for one part's stage-3 output.

    Flat-keyed rather than nested by model first: every table in this report pools across
    models before splitting by stratum, so the pair key only needs to keep two items from
    different models from colliding.
    """
    out: dict[str, dict[str, dict]] = {}
    for path in sorted((results / directory).glob("stage3_*.jsonl")):
        for row in _rows(path):
            key = f"{row['model']}::{row['item_id']}"
            out.setdefault(key, {})[row["condition"]] = row
    return out


def item_attribute(conds: dict[str, dict]) -> str | None:
    """The named attribute for one item, read off whichever condition row is present -- the
    field is written identically onto every condition of a given item, so any row will do."""
    for row in conds.values():
        attr = row.get("attribute")
        if attr is not None:
            return attr
    return None


def is_parentage(attribute: str | None) -> bool:
    return attribute in PARENTAGE_ATTRIBUTES


def split_by_stratum(items: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict[str, dict]]]:
    """Partition items into the parentage-named and non-parentage-named strata, plus the
    pooled union as a check that the two strata reproduce the unsplit numbers."""
    parentage: dict[str, dict[str, dict]] = {}
    non_parentage: dict[str, dict[str, dict]] = {}
    for key, conds in items.items():
        (parentage if is_parentage(item_attribute(conds)) else non_parentage)[key] = conds
    return {"parentage": parentage, "non_parentage": non_parentage, "pooled": items}


def model_names(items: dict[str, dict[str, dict]]) -> list[str]:
    models = {row["model"] for conds in items.values() for row in conds.values() if "model" in row}
    return sorted(short_model(m) for m in models)


# --------------------------------------------------------------------------------- measures


def discrete_outcomes(items: dict[str, dict[str, dict]]) -> dict[str, dict[str, bool | None]]:
    return {
        key: {c: row.get("chosen_is_edited") for c, row in conds.items()}
        for key, conds in items.items()
    }


def discrete_contrast(items: dict[str, dict[str, dict]], a: str, b: str) -> dict:
    """McNemar b/c, exact p, and the matched-pairs odds ratio for one contrast on one item set."""
    outcomes = discrete_outcomes(items)
    mc = mcnemar(outcomes, a, b)
    orr = odds_ratio(mc["b_a_only"], mc["c_b_only"])
    return {**mc, "or": orr, "contrast": f"{a}-{b}"}


def continuous_pairs(items: dict[str, dict[str, dict]], a: str, b: str) -> list[float]:
    """Per-item `delta_p_edited` at `a` minus at `b`, complete-case: both conditions must have a
    non-null `delta_p_edited` and both rows' `letter_probe.complete` must be true."""
    diffs = []
    for conds in items.values():
        ra, rb = conds.get(a), conds.get(b)
        if ra is None or rb is None:
            continue
        pa, pb = ra.get("delta_p_edited"), rb.get("delta_p_edited")
        if pa is None or pb is None:
            continue
        if not (ra.get("letter_probe") or {}).get("complete"):
            continue
        if not (rb.get("letter_probe") or {}).get("complete"):
            continue
        diffs.append(pa - pb)
    return diffs


def continuous_contrast(items: dict[str, dict[str, dict]], a: str, b: str,
                        *, seed: int = SEED) -> dict:
    diffs = continuous_pairs(items, a, b)
    result = bootstrap_ci_mean_diff(diffs, seed=seed, n_resamples=RESAMPLES)
    result["contrast"] = f"{a}-{b}"
    return result


# --------------------------------------------------------------------------------- reporting


def _fmt_ci(d: dict, places: int = 4) -> str:
    if d.get("ci_low") is None:
        return "n/a"
    return f"{d['mean']:+.{places}f} [{d['ci_low']:+.{places}f}, {d['ci_high']:+.{places}f}]"


def _fmt_or(o: dict) -> str:
    return f"{o['or']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}]{'*' if o['haldane_anscombe'] else ''}"


def analyze_part(results: Path, part_key: str) -> dict:
    directory, label = PARTS[part_key]
    items = load_part(results, directory)
    strata = split_by_stratum(items)

    out: dict = {
        "directory": directory,
        "label": label,
        "models": model_names(items),
        "stratum_sizes": {s: len(strata[s]) for s in STRATA},
        "contrasts": {},
    }
    for a, b in CONTRASTS:
        contrast_key = f"{a}-{b}"
        out["contrasts"][contrast_key] = {}
        for stratum in STRATA:
            its = strata[stratum]
            d = discrete_contrast(its, a, b)
            c = continuous_contrast(its, a, b)
            out["contrasts"][contrast_key][stratum] = {
                "n_items": len(its),
                "discrete": {
                    "b": d["b_a_only"], "c": d["c_b_only"], "n_discordant": d["n_discordant"],
                    "p_exact_two_sided": d["p_exact_two_sided"],
                    "or": d["or"]["or"], "or_ci_low": d["or"]["lo"], "or_ci_high": d["or"]["hi"],
                    "haldane_anscombe": d["or"]["haldane_anscombe"],
                },
                "continuous": {
                    "mean": c["mean"], "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                    "n": c["n"],
                },
            }
    return out


def build_text_report(results: dict[str, dict]) -> str:
    lines: list[str] = []
    w = lines.append

    w("# Relation-stratum check on the content contrasts")
    w("")
    w(f"Seed {SEED}, {RESAMPLES} bootstrap resamples. Parentage attributes: "
      f"{sorted(PARENTAGE_ATTRIBUTES)}.")
    w("")

    for part_key, part in results.items():
        w(f"## {part['label']} ({part['directory']})")
        w("")
        w(f"Models: {', '.join(part['models'])}")
        w("")
        sizes = part["stratum_sizes"]
        w(f"Stratum sizes (unique model x item_id pairs): "
          f"parentage-named = {sizes['parentage']}, "
          f"non-parentage-named = {sizes['non_parentage']}, "
          f"pooled = {sizes['pooled']}.")
        if sizes["parentage"] < 100:
            w("The parentage-named stratum is small. A null there is underpowered, not")
            w("evidence of absence; only a surviving effect in this stratum is informative on")
            w("its own. Absence of significance here must not be read as absence of an effect.")
        w("")
        w("| contrast | stratum | n items | b | c | n disc | exact p | OR [95% CI] "
          "| n cont | mean delta_p [95% CI] |")
        w("|---|---|---:|---:|---:|---:|---:|---|---:|---|")
        for contrast_key, strata in part["contrasts"].items():
            for stratum in STRATA:
                s = strata[stratum]
                disc, cont = s["discrete"], s["continuous"]
                or_str = (f"{disc['or']:.2f} [{disc['or_ci_low']:.2f}, {disc['or_ci_high']:.2f}]"
                          f"{'*' if disc['haldane_anscombe'] else ''}")
                cont_str = ("n/a" if cont["ci_low"] is None else
                            f"{cont['mean']:+.4f} [{cont['ci_low']:+.4f}, {cont['ci_high']:+.4f}]")
                w(f"| {contrast_key} | {STRATUM_LABEL[stratum]} | {s['n_items']} "
                  f"| {disc['b']} | {disc['c']} | {disc['n_discordant']} "
                  f"| {disc['p_exact_two_sided']:.4f} | {or_str} "
                  f"| {cont['n']} | {cont_str} |")
        w("")
    w("`*` marks an odds ratio computed with a Haldane-Anscombe +0.5 correction because a")
    w("discordant cell was empty.")
    w("")
    return "\n".join(lines) + "\n"


def build_summary(results: dict[str, dict]) -> dict:
    return {
        "seed": SEED,
        "n_resamples": RESAMPLES,
        "parentage_attributes": sorted(PARENTAGE_ATTRIBUTES),
        "contrasts": [f"{a}-{b}" for a, b in CONTRASTS],
        "strata": STRATA,
        "parts": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=REPO_ROOT / "code" / "results", type=Path,
                    help="results tree holding exp3a and exp3c (default: the repo's "
                         "code/results directory)")
    ap.add_argument("--part", choices=["a", "c", "both"], default="both",
                    help="which part to analyze (default: both)")
    ap.add_argument("--out", default=None, type=Path,
                    help="where to write the machine-readable summary "
                         "(default: code/results/relation_stratum_summary.json)")
    args = ap.parse_args(argv)

    part_keys = ["a", "c"] if args.part == "both" else [args.part]
    results = {part_key.upper(): analyze_part(args.results, part_key) for part_key in part_keys}

    print(build_text_report(results))

    out = args.out or (REPO_ROOT / "code" / "results" / "relation_stratum_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_summary(results), indent=2, sort_keys=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
