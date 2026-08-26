"""Offline audit of whether the third-option content effect is a function of the third option's
own pre-edit baseline, computed from the committed stage-3 JSONL alone.

No GPU, no network, no model, no new generation: every number here is a deterministic function
of files already in ``code/results/``. Four questions:

    1. How far apart do the rival and the third option start (R0), paired per item?
    2. Does the R3-R4 effect at the third option track the third option's own R0 baseline?
    3. Does the R1-R2 effect at the rival track the rival's own R0 baseline, on like terms?
    4. Conditioned on how close the two locations start (the tight and loose baseline-separation
       bands defined below), what do the location contrasts (R1-R3, R2-R4) look like within
       each band?

    python -m pilot.audit_location_confound

Reuses ``analyze_run3``'s loaders (``load_part``), its discrete- and continuous-contrast
plumbing (``discrete_outcomes``, ``discrete_contrast``, ``paired_mean``), and
``run_experiment.bootstrap_ci_mean_diff`` rather than reimplementing any of it. scipy is not a
dependency of this project (see ``code/requirements.txt``), so the correlation test in sections 2
and 3 is a seeded permutation test, the same discipline ``audit_probe_missingness.py`` and
``audit_position_bias.py`` already use for their own permutation tests.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from .analyze_run3 import (
    PARTS,
    discrete_contrast,
    discrete_outcomes,
    load_part,
    paired_mean,
    short_model,
)
from .config import REPO_ROOT, now_utc, stamp_utc
from .run_experiment import bootstrap_ci_mean_diff

# Same seed the rest of this project's offline audits use.
SEED = 20260822
RESAMPLES = 5_000
BOOTSTRAP_RESAMPLES = 10_000

# Baseline-separation bands, carried over unchanged from this project's earlier feasibility pass
# so the two are comparable: a floor on both baseline probabilities (so a near-zero denominator
# cannot produce an extreme ratio), then a ratio band around 1 at two widths. The earlier counts
# came from a script that was never committed, so this module documents its own population
# instead of trying to reproduce theirs.
BAND_FLOOR = 0.01
TIGHT_BAND = (0.67, 1.5)
LOOSE_BAND = (0.33, 3.0)

# A discordant count (McNemar's b+c) below this is reported but not interpreted -- the same
# threshold and rationale audit_position_bias.py uses.
TOO_SMALL_DISCORDANT = 5
# Fewer complete-case pairs than this: report n and skip the bin table rather than draw a
# quartile split from a handful of points.
TOO_SMALL_TO_BIN = 8
N_BINS = 4

LOCATION_CONTRASTS = [("R1", "R3", "location, relevant content"),
                      ("R2", "R4", "location, irrelevant content")]


def pooled(models: dict[str, dict[str, dict[str, dict]]]) -> dict[str, dict[str, dict]]:
    return {f"{m}::{i}": conds for m, its in models.items() for i, conds in its.items()}


# ------------------------------------------------------------------- section 1: baseline separation


def location_baselines(items: dict[str, dict[str, dict]]) -> dict[str, tuple[float, float]]:
    """``{item_id: (rival_R0_prob, third_R0_prob)}``, paired per item.

    Read off R0's own ``letter_probe.probs`` for R1's and R3's ``edited_letter`` -- the same
    population ``analyze_run3.baseline_by_target_paper`` uses and the paper's own baseline table
    reports, gated only on the letter appearing in R0's own ``probs`` (not on either row's own
    probe completeness). An item enters this dict only if both R1 and R3 exist and both letters
    are present in R0's read.
    """
    out: dict[str, tuple[float, float]] = {}
    for item_id, conds in items.items():
        r0 = conds.get("R0")
        if r0 is None:
            continue
        probs = (r0.get("letter_probe") or {}).get("probs") or {}
        r1, r3 = conds.get("R1"), conds.get("R3")
        if r1 is None or r3 is None:
            continue
        rl, tl = r1.get("edited_letter"), r3.get("edited_letter")
        if rl is None or tl is None or rl not in probs or tl not in probs:
            continue
        out[item_id] = (probs[rl], probs[tl])
    return out


def summarize_baseline_pairs(pairs: dict[str, tuple[float, float]]) -> dict:
    """Paired rival-minus-third baseline separation: n, means, bootstrap CI on the paired
    difference, and the share of items where the third option starts below the rival."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "mean_rival": None, "mean_third": None, "diff_ci": None,
                "below": 0, "share_below": None}
    rivals = [r for r, _ in pairs.values()]
    thirds = [t for _, t in pairs.values()]
    diffs = [r - t for r, t in pairs.values()]
    below = sum(1 for r, t in pairs.values() if t < r)
    diff_ci = bootstrap_ci_mean_diff(diffs, seed=SEED, n_resamples=BOOTSTRAP_RESAMPLES)
    return {"n": n, "mean_rival": sum(rivals) / n, "mean_third": sum(thirds) / n,
            "diff_ci": diff_ci, "below": below, "share_below": below / n}


# ------------------------------------------------------------- sections 2 and 3: dose response


def dose_response_series(
    items: dict[str, dict[str, dict]], a: str, b: str, kind: str,
) -> list[tuple[str, float, float]]:
    """``(item_id, baseline, outcome)`` triples for the ``a``-``b`` contrast, complete-case.

    ``baseline`` is condition ``a``'s own ``r0_p_target`` -- R0's read on the letter ``a`` edits,
    gated (per ``run_experiment.run_stage3``) on both R0 and ``a``'s own letter probe being
    complete. ``kind='discrete'`` reads ``chosen_is_edited`` on both sides (the outcome McNemar
    consumes); ``kind='continuous'`` reads ``delta_p_edited`` on both sides (the baseline-corrected
    probability change ``analyze_run3.paired_mean`` already uses). A row missing any of the three
    values needed is excluded, never imputed.
    """
    rows: list[tuple[str, float, float]] = []
    for item_id, conds in items.items():
        ca, cb = conds.get(a), conds.get(b)
        if ca is None or cb is None:
            continue
        baseline = ca.get("r0_p_target")
        if baseline is None:
            continue
        if kind == "discrete":
            va, vb = ca.get("chosen_is_edited"), cb.get("chosen_is_edited")
            if va is None or vb is None:
                continue
            outcome = float(bool(va)) - float(bool(vb))
        elif kind == "continuous":
            va, vb = ca.get("delta_p_edited"), cb.get("delta_p_edited")
            if va is None or vb is None:
                continue
            outcome = va - vb
        else:
            raise ValueError(f"unknown kind {kind!r}")
        rows.append((item_id, baseline, outcome))
    return rows


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient, or ``None`` if fewer than two points or either side is
    constant (zero variance makes the coefficient undefined, not zero)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def permutation_corr_test(
    xs: list[float], ys: list[float], *, seed: int = SEED, resamples: int = RESAMPLES,
) -> dict:
    """Seeded permutation test of Pearson r against a null of no association: ``ys`` is
    reshuffled ``resamples`` times (breaking the pairing while preserving each side's own
    distribution) and the p-value is the fraction of resamples at least as extreme in magnitude
    as the observed r, with the same ``+1``/``+1`` correction the rest of this project's
    permutation tests use so a p-value of exactly 0 is never reported from a finite resample."""
    n = len(xs)
    r_obs = pearson_r(xs, ys)
    if r_obs is None:
        return {"r": None, "p_value": None, "n": n, "resamples": resamples, "seed": seed}
    rng = random.Random(seed)
    shuffled = list(ys)
    at_least = 0
    for _ in range(resamples):
        rng.shuffle(shuffled)
        r = pearson_r(xs, shuffled)
        if r is not None and abs(r) >= abs(r_obs) - 1e-12:
            at_least += 1
    return {"r": r_obs, "p_value": (at_least + 1) / (resamples + 1), "n": n,
            "resamples": resamples, "seed": seed}


def quantile_bin_means(xs: list[float], ys: list[float], n_bins: int = N_BINS) -> list[dict]:
    """Split items into ``n_bins`` (fewer if ``n`` is small) equal-sized groups by ``xs``
    (sorted ascending), reporting each bin's baseline range, mean baseline, mean outcome, and n.

    Returns ``[]`` if there are fewer than ``TOO_SMALL_TO_BIN`` points -- a quartile split of a
    handful of points is not a trend a reader can see, so it is not drawn rather than drawn and
    mis-read. Bin count itself shrinks (never grows past what ``n_bins`` asks for) so that no bin
    has fewer than about 5 points by construction once there are enough points to bin at all.
    """
    n = len(xs)
    if n < TOO_SMALL_TO_BIN:
        return []
    bins = max(1, min(n_bins, n // 5))
    order = sorted(range(n), key=lambda i: xs[i])
    base, rem = divmod(n, bins)
    out: list[dict] = []
    idx = 0
    for b in range(bins):
        size = base + (1 if b < rem else 0)
        members = order[idx: idx + size]
        idx += size
        if not members:
            continue
        bx = [xs[i] for i in members]
        by = [ys[i] for i in members]
        out.append({"n": len(members), "baseline_lo": min(bx), "baseline_hi": max(bx),
                    "mean_baseline": sum(bx) / len(bx), "mean_outcome": sum(by) / len(by)})
    return out


# ------------------------------------------------------------ section 4: conditioned on baseline gap


def band_membership(rival: float, third: float) -> set[str]:
    """Which of ``{"tight", "loose"}`` this item's rival/third pair qualifies for.

    Both bands require ``min(rival, third) >= BAND_FLOOR`` (a near-zero baseline on either side
    makes the ratio arbitrarily extreme without the two starting points actually being far
    apart in any practical sense). The tight band nests inside the loose one by construction
    (``TIGHT_BAND`` is a subset of ``LOOSE_BAND``), so every tight item is also a loose item.
    """
    if third == 0 or min(rival, third) < BAND_FLOOR:
        return set()
    ratio = rival / third
    membership = set()
    if TIGHT_BAND[0] <= ratio <= TIGHT_BAND[1]:
        membership.add("tight")
    if LOOSE_BAND[0] <= ratio <= LOOSE_BAND[1]:
        membership.add("loose")
    return membership


def items_in_band(baselines: dict[str, tuple[float, float]], band: str) -> set[str]:
    return {item_id for item_id, (r, t) in baselines.items() if band in band_membership(r, t)}


def continuous_paired_mean_delta_p(
    items: dict[str, dict[str, dict]], a: str, b: str,
) -> dict:
    """``paired_mean`` on ``delta_p_edited`` -- the same one-liner ``analyze_run3.build_report``
    builds inline for its own audit table, reused here rather than re-derived."""
    cont = {
        item_id: {c: {"delta_p_edited": row.get("delta_p_edited")} for c, row in conds.items()}
        for item_id, conds in items.items()
    }
    return paired_mean(cont, a, b, "delta_p_edited")


# --------------------------------------------------------------------------------- reporting


def _ci(d: dict | None, places: int = 4) -> str:
    if d is None or d.get("ci_low") is None:
        return "n/a"
    return f"{d['mean']:+.{places}f} [{d['ci_low']:+.{places}f}, {d['ci_high']:+.{places}f}]"


def _fmt(x: float | None, places: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def build_report(results: Path) -> str:
    lines: list[str] = []
    w = lines.append

    w("# Third-option baseline as a confound for the location contrast")
    w("")
    w(f"Generated by `python -m pilot.audit_location_confound` at {stamp_utc(now_utc())}, from")
    w("the committed stage-3 JSONL in `code/results/exp3a`, `exp3b`, `exp3c`. No GPU, no network,")
    w("no generation. Every permutation test uses seed `{}` with `{}` resamples; every bootstrap".format(
        SEED, RESAMPLES))
    w(f"CI uses seed `{SEED}` with `{BOOTSTRAP_RESAMPLES}` resamples, the same discipline this")
    w("project's other offline audits use. scipy is not a dependency of this project (checked")
    w("against `code/requirements.txt`), so every correlation test here is a permutation test over")
    w("Pearson r rather than a library p-value.")
    w("")

    loaded = {part: load_part(results, part) for part in PARTS}

    # --------------------------------------------------------------------------- section 1
    w("## 1. Baseline separation: rival vs. third option, paired per item (R0)")
    w("")
    w("Each item's R0 read on the rival's own letter and on the third option's own letter,")
    w("paired within the item, not just the two marginal means already reported elsewhere.")
    w("")
    w("| part | scope | n | mean rival | mean third | paired diff (rival-third) [95% CI] "
      "| n third < rival | share |")
    w("|---|---|---:|---:|---:|---|---:|---:|")
    baselines_by_part: dict[str, dict[str, tuple[float, float]]] = {}
    for part in PARTS:
        models = loaded[part]
        pooled_items = pooled(models)
        scopes = [(short_model(m), its) for m, its in sorted(models.items())]
        scopes.append(("**pooled**", pooled_items))
        for scope, its in scopes:
            bl = location_baselines(its)
            if scope == "**pooled**":
                baselines_by_part[part] = bl
            s = summarize_baseline_pairs(bl)
            share = "n/a" if s["share_below"] is None else f"{s['share_below']:.3f}"
            w(f"| {part} | {scope} | {s['n']} | {_fmt(s['mean_rival'])} | {_fmt(s['mean_third'])} "
              f"| {_ci(s['diff_ci'], 4)} | {s['below']} | {share} |")
    w("")
    w("`share` is the fraction of paired items where the third option's R0 read is below the")
    w("rival's. A rate at or near 0 or 1 here is expected from the design (gate 8 selects the")
    w("third option partly for already lacking the named attribute) and is reported, not treated")
    w("as a defect in this audit.")
    w("")

    # --------------------------------------------------------------------------- section 2
    w("## 2. Dose-response: the third-option effect (R3-R4) against its own R0 baseline")
    w("")
    w("Per item, the R3-R4 outcome (content added at the third option) against that same item's")
    w("R3 `r0_p_target` -- the third option's own pre-edit probability. If the third-option effect")
    w("is mostly a function of how little room there was to begin with, this correlation should")
    w("be strong and the bin means should trend monotonically; if the effect is independent of")
    w("the starting point, it should not.")
    w("")
    _dose_response_section(w, loaded, "R3", "R4", "third option's own")

    # --------------------------------------------------------------------------- section 3
    w("## 3. Dose-response: the rival effect (R1-R2) against its own R0 baseline, on like terms")
    w("")
    w("The same test at the location the model actually named, so the two locations are compared")
    w("on identical machinery rather than testing one and leaving the other as a bare assertion.")
    w("")
    _dose_response_section(w, loaded, "R1", "R2", "rival's own")

    # --------------------------------------------------------------------------- section 4
    w("## 4. The location contrast, conditioned on how close the two locations start")
    w("")
    w(f"Bands carried over from the earlier feasibility pass: both baselines at or above "
      f"`{BAND_FLOOR}`, then a ratio band around 1 -- tight `{TIGHT_BAND[0]}-{TIGHT_BAND[1]}`,")
    w(f"loose `{LOOSE_BAND[0]}-{LOOSE_BAND[1]}` (tight items are a subset of loose items). Band")
    w("membership is computed from this module's own `location_baselines` pairing (section 1's")
    w("population); it need not reproduce the earlier item counts, since no committed script")
    w("preserves the method that produced them -- only the band definitions are reused.")
    w("")
    for part in PARTS:
        models = loaded[part]
        pooled_items = pooled(models)
        bl_pooled = baselines_by_part[part]
        scopes = [(short_model(m), its, location_baselines(its)) for m, its in sorted(models.items())]
        scopes.append(("**pooled**", pooled_items, bl_pooled))
        w(f"### Part {part}")
        w("")
        for band in ("tight", "loose"):
            w(f"**{band} band**")
            w("")
            w("| scope | n in band | contrast | n disc | b | c | exact p | RD [95% CI] "
              "| n cont | Δp [95% CI] | note |")
            w("|---|---:|---|---:|---:|---:|---:|---|---:|---|---|")
            for scope, its, bl in scopes:
                band_ids = items_in_band(bl, band)
                band_items = {i: c for i, c in its.items() if i in band_ids}
                disc = discrete_outcomes(band_items)
                for a, b, _why in LOCATION_CONTRASTS:
                    d = discrete_contrast(disc, a, b)
                    cm = continuous_paired_mean_delta_p(band_items, a, b)
                    too_small = (d["b_a_only"] + d["c_b_only"]) < TOO_SMALL_DISCORDANT
                    note = "too small to read" if too_small else ""
                    w(f"| {scope} | {len(band_ids)} | {a}-{b} | {d['n_paired']} "
                      f"| {d['b_a_only']} | {d['c_b_only']} | {d['p_exact_two_sided']:.4f} "
                      f"| {_ci(d['rd'], 3)} | {cm['n']} | {_ci(cm)} | {note} |")
            w("")
    w(f"A row flagged `too small to read` has fewer than {TOO_SMALL_DISCORDANT} discordant pairs")
    w("(McNemar's b+c); its point estimate is reported above but not interpreted.")
    w("")
    return "\n".join(lines) + "\n"


def _dose_response_section(w, loaded: dict, a: str, b: str, baseline_label: str) -> None:
    w(f"**Correlation of the {a}-{b} outcome with the {baseline_label} R0 baseline**")
    w("")
    w("| part | scope | kind | n | Pearson r | permutation p |")
    w("|---|---|---|---:|---:|---:|")
    series_by_part: dict[str, dict[str, dict[str, list]]] = {}
    for part in PARTS:
        models = loaded[part]
        pooled_items = pooled(models)
        scopes = [(short_model(m), its) for m, its in sorted(models.items())]
        scopes.append(("**pooled**", pooled_items))
        series_by_part[part] = {}
        for scope, its in scopes:
            series_by_part[part][scope] = {}
            for kind in ("discrete", "continuous"):
                rows = dose_response_series(its, a, b, kind)
                series_by_part[part][scope][kind] = rows
                xs = [r[1] for r in rows]
                ys = [r[2] for r in rows]
                test = permutation_corr_test(xs, ys)
                r_text = "n/a" if test["r"] is None else f"{test['r']:+.4f}"
                p_text = "n/a" if test["p_value"] is None else f"{test['p_value']:.4f}"
                w(f"| {part} | {scope} | {kind} | {test['n']} | {r_text} | {p_text} |")
    w("")
    w(f"**Bin means, pooled per part** (baseline quartiles, up to {N_BINS} bins; fewer than "
      f"{TOO_SMALL_TO_BIN} complete-case pairs is reported as n only, not binned)")
    w("")
    for kind in ("discrete", "continuous"):
        w(f"*{kind}*")
        w("")
        w("| part | bin | n | baseline range | mean baseline | mean outcome |")
        w("|---|---:|---:|---|---:|---:|")
        for part in PARTS:
            rows = series_by_part[part]["**pooled**"][kind]
            xs = [r[1] for r in rows]
            ys = [r[2] for r in rows]
            bins = quantile_bin_means(xs, ys)
            if not bins:
                w(f"| {part} | - | {len(xs)} | n/a (fewer than {TOO_SMALL_TO_BIN} pairs) | n/a | n/a |")
                continue
            for i, bin_ in enumerate(bins, start=1):
                rng_text = f"[{bin_['baseline_lo']:.4f}, {bin_['baseline_hi']:.4f}]"
                w(f"| {part} | {i} | {bin_['n']} | {rng_text} | {bin_['mean_baseline']:.4f} "
                  f"| {bin_['mean_outcome']:+.4f} |")
        w("")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=Path("results"), type=Path,
                     help="the results tree holding exp3a, exp3b, exp3c (default: results)")
    ap.add_argument("--out", default=None, type=Path,
                     help="where to write the report (default: LOCATION_CONFOUND_AUDIT.md at the "
                          "tree root)")
    args = ap.parse_args(argv)

    # Report lands at the tree root; this project keeps its write-ups outside the code tree, so
    # the generated file is moved there afterwards rather than written across trees from here.
    out = args.out or (REPO_ROOT / "LOCATION_CONFOUND_AUDIT.md")
    report = build_report(args.results)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
