"""Offline analysis of E3 (necessity), computed from committed `stage_necessity_*.jsonl` alone.

No GPU, no network, no new generation. Reuses the project's own statistics rather than
reimplementing them: `odds_ratio`, `discrete_contrast`, `paired_mean`, `bootstrap_stat`, `holm`,
`mcnemar_power`, `risk_difference`, `short_model`, `SEED`/`RESAMPLES` come from
`pilot.analyze_run3`; `leave_one_model_out`, `heterogeneity`, `cluster_bootstrap_mean_diff`,
`stratified_contrast`, `_pooled_discrete`, `_pooled_continuous` come from `pilot.analyze_round5`.

Those helpers are written against run 3's `{model: {item_id: {condition: row}}}` shape with two
field names hardcoded in several of them (`chosen_is_edited`, `delta_p_edited`). E3's own fields
are named differently (`still_chooses_original`, `delta_p_original`) and its conditions are
N0/N1/N2, not R0-R4 -- so rather than editing those hardcoded field names, `_adapt_for_round5`
below re-keys E3's rows onto the names those helpers already expect, once, and every reused
function then runs completely unmodified.

    python -m pilot.analyze_necessity --self-check                    # arithmetic only, no files
    python -m pilot.analyze_necessity                                 # everything found under
                                                                        # --results-dir
    python -m pilot.analyze_necessity --section discrete
    python -m pilot.analyze_necessity --section e1-estimate           # GPU-cost projection only

Primary contrast: N1 vs N2 (does removing the model's own stated reason move it off its choice,
against a length-matched control deletion). Secondary: N1 vs N0. Both are computed only over
items whose N0 (unedited) re-ask reproduced the original stage-1 choice -- the same restriction
`run_necessity.necessity_contrasts` applies, recomputed independently here from the committed
rows rather than imported, since this module reads files a run already wrote rather than
re-running anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Callable

from . import config as C
from .analyze_round5 import (
    _pooled_continuous,  # noqa: F401 -- reused via leave_one_model_out/heterogeneity
    _pooled_discrete,  # noqa: F401 -- reused via leave_one_model_out/heterogeneity
    cluster_bootstrap_mean_diff,
    heterogeneity,
    leave_one_model_out,
    stratified_contrast,
)
from .analyze_run3 import (
    RESAMPLES,
    SEED,
    bootstrap_stat,
    discrete_contrast,
    mcnemar_power,
    paired_mean,
    short_model,
)
from .run_experiment import bootstrap_ci_mean_diff, mcnemar  # noqa: F401 -- reused throughout
from .watchdog import LEDGER_FIELDS

CONDITION_LABELS = ("N0", "N1", "N2")
CONTRASTS = [("N1", "N2", "primary: control-deletion baseline"),
             ("N1", "N0", "secondary: unedited baseline")]

# Below this many discordant pairs, McNemar's own exact test is reporting on almost nothing --
# same discipline as analyze_round5.FLUENCY_LOO_MIN_DISCORDANT, applied here to the strata
# section, which is where this project's own committed splits are most likely to produce one.
MIN_DISCORDANT_FOR_INFERENCE = 10


# ============================================================================== loading


def _rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_necessity_rows(results_dir: Path) -> list[dict]:
    """Every row of every `stage_necessity_*.jsonl` under `results_dir`, recursive -- E3 writes
    flat into whatever `--out-dir` a run used, not into a per-part subdirectory the way run 3's
    stage files do, so a shallow glob would miss a run pointed at a subdirectory."""
    rows: list[dict] = []
    for path in sorted(results_dir.glob("**/stage_necessity_*.jsonl")):
        rows.extend(_rows(path))
    return rows


def build_nested(rows: list[dict]) -> dict[str, dict[str, dict[str, dict]]]:
    """`{model: {item_id: {condition: row}}}`, the shape every reused helper expects. A repeated
    `(model, item_id, condition)` key (a re-run appending to the same file) keeps the last row
    written, same last-write-wins convention `analyze_round5.load_surprisal` uses."""
    out: dict[str, dict[str, dict[str, dict]]] = {}
    for row in rows:
        out.setdefault(row["model"], {}).setdefault(row["item_id"], {})[row["condition"]] = row
    return out


def interpretable(nested: dict) -> dict[str, dict[str, dict[str, dict]]]:
    """Only the (model, item) pairs whose N0 row reproduced the original choice
    (`still_chooses_original is True`) -- an N0 flip means nothing else about that item is
    interpretable, mirroring `run_necessity.run_stage_necessity`'s own docstring on why N0 is an
    integrity check, not a treatment."""
    out: dict[str, dict[str, dict[str, dict]]] = {}
    for model, items in nested.items():
        kept = {
            item_id: conds for item_id, conds in items.items()
            if (conds.get("N0") or {}).get("still_chooses_original") is True
        }
        if kept:
            out[model] = kept
    return out


def _discrete_outcomes(items: dict[str, dict[str, dict]]) -> dict[str, dict[str, bool | None]]:
    """`still_chooses_original` per item per condition -- the shape `mcnemar`/`discrete_contrast`
    consume directly, so no field renaming is needed for this half of the analysis."""
    return {
        item_id: {c: row.get("still_chooses_original") for c, row in conds.items()}
        for item_id, conds in items.items()
    }


def _continuous_outcomes(items: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """`{item_id: {condition: {"delta_p_original": value}}}` -- `paired_mean` takes the field
    name as an explicit argument, so `delta_p_original` is passed straight through with no
    renaming either."""
    return {
        item_id: {c: {"delta_p_original": row.get("delta_p_original")} for c, row in conds.items()}
        for item_id, conds in items.items()
    }


def _pooled(items_by_model: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    return {f"{m}::{i}": conds for m, its in sorted(items_by_model.items()) for i, conds in its.items()}


def _adapt_for_round5(
    nested: dict, *, attribute_fn: Callable[[dict], str | None] = lambda row: None,
) -> dict:
    """`nested`, re-keyed so `analyze_round5`'s helpers -- which hardcode `chosen_is_edited` and
    `delta_p_edited` internally rather than taking a field name -- read E3's `still_chooses_
    original`/`delta_p_original` under those names, plus an `attribute` key populated by
    `attribute_fn` for `stratified_contrast`'s own relation-bucketing to split on instead of the
    named attribute it was built for. Every original field is kept alongside the renamed ones."""
    out: dict[str, dict[str, dict[str, dict]]] = {}
    for model, items in nested.items():
        out[model] = {}
        for item_id, conds in items.items():
            out[model][item_id] = {
                cond: {
                    **row,
                    "chosen_is_edited": row.get("still_chooses_original"),
                    "delta_p_edited": row.get("delta_p_original"),
                    "attribute": attribute_fn(row),
                }
                for cond, row in conds.items()
            }
    return out


def _item_clusters(
    items_by_model: dict[str, dict[str, dict]], a: str, b: str, field: str, *, discrete: bool,
) -> dict[str, list[float]]:
    """`{item_id: [per-model diffs]}` for `cluster_bootstrap_mean_diff` -- the real dataset item
    id, not the `model::item_id` pooling key, so an item several models both built contributes
    several diffs to the same cluster. A thin, E3-specific counterpart of `analyze_round5`'s
    `_cluster_diffs_continuous`/`_cluster_diffs_discrete`, neither of which is in this module's
    reuse list (only the resampler `cluster_bootstrap_mean_diff` itself is) and both of which
    hardcode run 3's own field names."""
    clusters: dict[str, list[float]] = {}
    for model in sorted(items_by_model):
        for item_id, conds in items_by_model[model].items():
            ra, rb = conds.get(a), conds.get(b)
            if ra is None or rb is None:
                continue
            if discrete:
                va, vb = ra.get(field), rb.get(field)
                if va is None or vb is None:
                    continue
                clusters.setdefault(item_id, []).append(float(bool(va)) - float(bool(vb)))
            else:
                va, vb = ra.get(field), rb.get(field)
                if va is None or vb is None:
                    continue
                clusters.setdefault(item_id, []).append(va - vb)
    return clusters


# ============================================================================== 1. integrity


def report_integrity(nested: dict) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {"per_model": {}, "pooled": {}}
    w("## 1. Integrity: does N0 reproduce the stage-1 choice?")
    w("")
    w("N0 is the unedited item, re-asked under the same greedy decoding as stage 1. It should")
    w("reproduce the original choice on every item; a nonzero flip count is a reportable finding")
    w("(a decoding or reconstruction inconsistency), not something to filter away quietly. Items")
    w("with an N0 flip are excluded from every contrast below, exactly as `run_necessity.")
    w("necessity_contrasts` excludes them, and that exclusion count is reported here.")
    w("")
    w("| model | n | reproduced | flipped | unreadable | flip rate |")
    w("|---|---:|---:|---:|---:|---:|")

    def _counts(items: dict[str, dict[str, dict]]) -> dict:
        n = reproduced = flipped = unreadable = 0
        for conds in items.values():
            n0 = conds.get("N0")
            if n0 is None:
                continue
            n += 1
            v = n0.get("still_chooses_original")
            if v is True:
                reproduced += 1
            elif v is False:
                flipped += 1
            else:
                unreadable += 1
        return {"n": n, "reproduced": reproduced, "flipped": flipped, "unreadable": unreadable,
                "flip_rate": (flipped / n) if n else None}

    pooled_items: dict[str, dict[str, dict]] = {}
    for model in sorted(nested):
        c = _counts(nested[model])
        payload["per_model"][short_model(model)] = c
        rate = "n/a" if c["flip_rate"] is None else f"{c['flip_rate']:.4f}"
        w(f"| {short_model(model)} | {c['n']} | {c['reproduced']} | {c['flipped']} "
          f"| {c['unreadable']} | {rate} |")
        for item_id, conds in nested[model].items():
            pooled_items[f"{model}::{item_id}"] = conds
    pc = _counts(pooled_items)
    payload["pooled"] = pc
    rate = "n/a" if pc["flip_rate"] is None else f"{pc['flip_rate']:.4f}"
    w(f"| **pooled** | {pc['n']} | {pc['reproduced']} | {pc['flipped']} | {pc['unreadable']} "
      f"| {rate} |")
    w("")
    if pc["flipped"]:
        w(f"**{pc['flipped']} of {pc['n']} item(s) flipped under N0 despite greedy, temperature-0")
        w("decoding on an unedited item.** This should not happen and is not filtered out of this")
        w("table; it is excluded only from the contrasts below, and its cause should be")
        w("investigated (a reconstruction mismatch, a non-deterministic backend setting, or a")
        w("genuinely non-deterministic model) before the contrasts are read as clean.")
    else:
        w("No N0 flips observed: every item's N0 re-ask reproduced its stage-1 choice.")
    w("")
    return "\n".join(lines) + "\n", payload


# ============================================================================== 2. discrete


def _discrete_rows(items_by_model: dict[str, dict[str, dict]], a: str, b: str) -> list[tuple[str, dict]]:
    rows = []
    for model in sorted(items_by_model):
        d = discrete_contrast(_discrete_outcomes(items_by_model[model]), a, b)
        rows.append((short_model(model), d))
    pooled = discrete_contrast(_discrete_outcomes(_pooled(items_by_model)), a, b)
    rows.append(("**pooled**", pooled))
    return rows


def report_discrete(nested: dict) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 2. Discrete contrast: still chooses the original option?")
    w("")
    w("Paired McNemar on `still_chooses_original`, per model and pooled, restricted to items")
    w("whose N0 reproduced the stage-1 choice. `or` is the matched-pairs odds ratio with a")
    w("log-scale Wald 95% CI; `rd` is the paired risk difference with a percentile bootstrap 95%")
    w(f"CI (seed `{SEED}`, `{RESAMPLES}` resamples). The last row resamples distinct item ids")
    w("(`cluster_bootstrap_mean_diff`) instead of rows, as a check on the pooled-row bootstrap:")
    w("an item several models both build contributes several rows, and the row-level bootstrap")
    w("above treats those as independent.")
    w("")
    items = interpretable(nested)
    for a, b, why in CONTRASTS:
        w(f"### {a} vs {b} ({why})")
        w("")
        w("| scope | n paired | b | c | exact p | OR [95% CI] | RD [95% CI] |")
        w("|---|---:|---:|---:|---:|---|---|")
        rows = _discrete_rows(items, a, b)
        payload[f"{a}-{b}"] = {scope: d for scope, d in rows}
        for scope, d in rows:
            o, rd = d["or"], d["rd"]
            rd_s = "n/a" if rd["ci_low"] is None else f"{rd['mean']:+.3f} [{rd['ci_low']:+.3f}, {rd['ci_high']:+.3f}]"
            w(f"| {scope} | {d['n_paired']} | {d['b_a_only']} | {d['c_b_only']} "
              f"| {d['p_exact_two_sided']:.4f} "
              f"| {o['or']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}]{'*' if o['haldane_anscombe'] else ''} "
              f"| {rd_s} |")
        clusters = _item_clusters(items, a, b, "still_chooses_original", discrete=True)
        cci = cluster_bootstrap_mean_diff(clusters, seed=SEED, n_resamples=RESAMPLES)
        payload[f"{a}-{b}"]["**pooled (item-clustered)**"] = cci
        cci_s = "n/a" if cci["ci_low"] is None else f"{cci['mean']:+.3f} [{cci['ci_low']:+.3f}, {cci['ci_high']:+.3f}]"
        w(f"| **pooled (item-clustered)** | {cci['n']} ({cci['n_clusters']} clusters) | -- | -- "
          f"| -- | -- | {cci_s} |")
        w("")
    w("`*` marks an odds ratio computed with a Haldane-Anscombe correction because a discordant")
    w("cell was empty.")
    w("")
    return "\n".join(lines) + "\n", payload


# ============================================================================== 3. continuous


def report_continuous(nested: dict) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 3. Continuous contrast: change in the original letter's probability")
    w("")
    w("Paired mean of `delta_p_original`, per model and pooled, over items complete on both")
    w("sides -- a *different*, usually smaller, population than section 2's: the letter probe")
    w("does not complete on every row, while the discrete read only needs a parseable free-text")
    w("answer. The two sections' `n` are never assumed to share a denominator.")
    w("")
    items = interpretable(nested)
    for a, b, why in CONTRASTS:
        w(f"### {a} vs {b} ({why})")
        w("")
        w("| scope | n (letter-probe complete) | mean delta p [95% CI] |")
        w("|---|---:|---|")
        rows = []
        for model in sorted(items):
            cm = paired_mean(_continuous_outcomes(items[model]), a, b, "delta_p_original", seed=SEED)
            rows.append((short_model(model), cm))
        pooled_cm = paired_mean(_continuous_outcomes(_pooled(items)), a, b, "delta_p_original", seed=SEED)
        rows.append(("**pooled**", pooled_cm))
        payload[f"{a}-{b}"] = {scope: cm for scope, cm in rows}
        for scope, cm in rows:
            ci = "n/a" if cm["ci_low"] is None else f"{cm['mean']:+.4f} [{cm['ci_low']:+.4f}, {cm['ci_high']:+.4f}]"
            w(f"| {scope} | {cm['n']} | {ci} |")
        clusters = _item_clusters(items, a, b, "delta_p_original", discrete=False)
        cci = cluster_bootstrap_mean_diff(clusters, seed=SEED, n_resamples=RESAMPLES)
        payload[f"{a}-{b}"]["**pooled (item-clustered)**"] = cci
        cci_s = "n/a" if cci["ci_low"] is None else f"{cci['mean']:+.4f} [{cci['ci_low']:+.4f}, {cci['ci_high']:+.4f}]"
        w(f"| **pooled (item-clustered)** | {cci['n']} ({cci['n_clusters']} clusters) | {cci_s} |")
        w("")
    w("Row-level and item-clustered pooled intervals agreeing is a check on the pooled-row")
    w("bootstrap used everywhere else in this report; a real disagreement between them would")
    w("mean an item that several models both build is pulling more weight than it should.")
    w("")
    return "\n".join(lines) + "\n", payload


# ============================================================================== 4. leave-one-out


def report_loo(nested: dict) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 4. Leave-one-model-out")
    w("")
    w("Both measures, pooled over every model and then again with each model dropped, for both")
    w("contrasts. Reuses `analyze_round5.leave_one_model_out` unchanged, over the round-5 field")
    w("names `_adapt_for_round5` maps E3's rows onto.")
    w("")
    adapted = _adapt_for_round5(interpretable(nested))
    for a, b, why in CONTRASTS:
        w(f"### {a} vs {b} ({why})")
        w("")
        w("| dropped | disc b | disc c | OR | exact p | cont n | delta p mean [95% CI] |")
        w("|---|---:|---:|---:|---:|---:|---|")
        rows = leave_one_model_out(adapted, a, b)
        payload[f"{a}-{b}"] = rows
        for row in rows:
            dropped = row["dropped"] or "-- (all models)"
            d, cm = row["discrete"], row["continuous"]
            ci = "n/a" if cm["ci_low"] is None else f"{cm['mean']:+.4f} [{cm['ci_low']:+.4f}, {cm['ci_high']:+.4f}]"
            w(f"| {dropped} | {d['b_a_only']} | {d['c_b_only']} | {d['or']['or']:.2f} "
              f"| {d['p_exact_two_sided']:.4f} | {cm['n']} | {ci} |")
        w("")
    return "\n".join(lines) + "\n", payload


# ============================================================================== 5. heterogeneity


def report_heterogeneity(nested: dict) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 5. Between-model heterogeneity")
    w("")
    w("Cochran's Q over the per-model log odds ratios, both contrasts. `tau^2` is not reported --")
    w("see `analyze_round5.heterogeneity`'s own docstring: with as few models as this project")
    w("runs, it would be estimated from almost no information. A three-model family cannot")
    w("support much heterogeneity inference at all; Q/I^2 here are a check for a gross outlier")
    w("model, not a precise variance estimate.")
    w("")
    w("| contrast | Q | df | p | I^2 |")
    w("|---|---:|---:|---:|---:|")
    adapted = _adapt_for_round5(interpretable(nested))
    for a, b, _why in CONTRASTS:
        h = heterogeneity(adapted, a, b)
        payload[f"{a}-{b}"] = h
        q = "n/a" if h["q"] is None else f"{h['q']:.2f}"
        i2 = "n/a" if h["i2"] is None else f"{h['i2']:.1f}%"
        p = "n/a" if h["p"] is None else f"{h['p']:.4f}"
        w(f"| {a}-{b} | {q} | {h['df']} | {p} | {i2} |")
    w("")
    return "\n".join(lines) + "\n", payload


# ============================================================================== 6. strata


def _strata_report(nested: dict, key: str, labels: tuple[str, str]) -> tuple[str, dict]:
    """One stratification of both contrasts by the boolean field `key`, reusing
    `stratified_contrast` with `min_rows=0` so both values of the boolean always get their own
    bucket rather than being folded into `stratified_contrast`'s own relation-sized `other`."""
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    true_label, false_label = labels
    attribute_fn = lambda row: true_label if row.get(key) else false_label  # noqa: E731
    adapted = _adapt_for_round5(interpretable(nested), attribute_fn=attribute_fn)
    w(f"### stratified on `{key}`")
    w("")
    w("| contrast | stratum | n rows | b | c | OR | exact p | too small for inference? |")
    w("|---|---|---:|---:|---:|---:|---:|---|")
    for a, b, _why in CONTRASTS:
        strat = stratified_contrast(adapted, a, b, min_rows=0)
        payload[f"{a}-{b}"] = strat
        for bucket_name in (true_label, false_label):
            s = strat.get(bucket_name)
            if s is None:
                w(f"| {a}-{b} | {bucket_name} | 0 | - | - | - | - | yes (no rows) |")
                continue
            d = s["discrete"]
            too_small = (d["b_a_only"] + d["c_b_only"]) < MIN_DISCORDANT_FOR_INFERENCE
            w(f"| {a}-{b} | {bucket_name} | {s['n_rows']} | {d['b_a_only']} | {d['c_b_only']} "
              f"| {d['or']['or']:.2f} | {d['p_exact_two_sided']:.4f} "
              f"| {'yes' if too_small else 'no'} |")
    w("")
    return "\n".join(lines) + "\n", payload


def report_strata(nested: dict) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 6. Stratification")
    w("")
    w("Deleting the named attribute can change which option is genuinely correct, so the")
    w("primary and secondary contrasts are split on the two fields E3 records for exactly this")
    w("purpose: whether the model's original choice was the gold answer, and whether the")
    w("question itself looks like an order/date comparison that could receive the deleted")
    w(f"attribute. A stratum with fewer than {MIN_DISCORDANT_FOR_INFERENCE} discordant pairs is")
    w("flagged rather than read as a null result.")
    w("")
    for key, labels, heading in (
        ("chosen_is_gold", ("gold", "not_gold"), None),
        ("question_relevant_attribute", ("relevant", "not_relevant"), None),
    ):
        text, sub_payload = _strata_report(nested, key, labels)
        lines.append(text)
        payload[key] = sub_payload
    return "\n".join(lines) + "\n", payload


# ============================================================================== 7. power


def report_power(nested: dict) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 7. Power at the observed discordant rate")
    w("")
    w("`mcnemar_power` (exact, by enumeration -- see `analyze_run3.mcnemar_power`) at the pooled")
    w("`n`, discordant rate and odds ratio this run actually observed, both contrasts. A p-value")
    w("above 0.05 next to a low power number here is an underpowered null, not evidence of no")
    w("effect; the same p-value next to a high power number is a more informative null.")
    w("")
    w("| contrast | n paired | discordant rate | OR | power at this n/rate/OR |")
    w("|---|---:|---:|---:|---:|")
    items = interpretable(nested)
    for a, b, _why in CONTRASTS:
        d = discrete_contrast(_discrete_outcomes(_pooled(items)), a, b)
        n = d["n_paired"]
        disc_rate = (d["b_a_only"] + d["c_b_only"]) / n if n else 0.0
        odds = d["or"]["or"]
        power = mcnemar_power(n, disc_rate, odds) if n and disc_rate > 0 else float("nan")
        payload[f"{a}-{b}"] = {"n": n, "disc_rate": disc_rate, "or": odds, "power": power}
        power_s = "n/a" if power != power else f"{power:.3f}"  # NaN != NaN
        w(f"| {a}-{b} | {n} | {disc_rate:.4f} | {odds:.2f} | {power_s} |")
    w("")
    return "\n".join(lines) + "\n", payload


# ============================================================================== 8. e1-estimate

# The measured 4-option built-item yield an E1-shaped run (stage 1 elicit -> stage 2 repair ->
# stage 3 re-ask/probe over R0-R4) actually produced against its own attempted pool: Part A,
# 387/1204 = 32.1% ("Pooled 387/1204 attempted (32.1%)", already published and audited for Part
# A). Not re-derived from a committed file here because it is Part A's own already-published,
# audited figure, not a number this module could recompute more authoritatively by re-reading
# exp3a's raw stage files.
E1_BUILT_YIELD_4OPT = 387 / 1204

E1_BUILT_TARGET_LOW = 300
E1_BUILT_TARGET_HIGH = 390
E1_CHECKPOINTS = 3
E1_CONDITIONS = 5  # R0-R4, the sufficiency design's own condition count

# ponytail: model-load time is not recoverable from the committed ledger at all -- the watchdog
# is constructed only after `get_backend` has already returned a loaded model (see
# `run_necessity.run_stage_necessity`, `run_experiment.run_stage3`), so every `gpu_seconds` this
# project has ever logged excludes load time by construction. `results/console.log`'s vLLM
# startup lines are not a substitute either: the gap between two "Loading model weights took"
# timestamps also contains that model's entire generation pass, not just the next model's load.
# Ceiling: the figure below is a documented assumption, not a measurement, and self_check checks
# only that it is used consistently, never that it is correct. Upgrade path: have the pod run
# log its own `time.monotonic()` immediately before and after `get_backend(...)` and read that
# instead, once it exists.
ASSUMED_MODEL_LOAD_S = 60.0

# round-6 run summaries this section prefers over the ledger-only estimate when present, since
# they would reflect the actual models/hardware this run used rather than run 3's history.
ROUND6_SUMMARY_FILES = ("necessity_summary.json", "probe_variants_summary.json")


def _verify_ledger_header(fieldnames: list[str] | None) -> None:
    """Rule 5: never trust a column name without checking the file has it. `watchdog.
    LEDGER_FIELDS` is the single place this project defines the ledger's schema; a mismatch here
    means the ledger format changed and every rate below would silently be reading garbage."""
    if fieldnames is None or set(LEDGER_FIELDS) - set(fieldnames):
        raise ValueError(
            f"gpu_ledger.csv header {fieldnames!r} is missing column(s) "
            f"{set(LEDGER_FIELDS) - set(fieldnames or [])}; expected {LEDGER_FIELDS}"
        )


def load_ledger_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _verify_ledger_header(reader.fieldnames)
        return list(reader)


def _rate_from_ledger(ledger_rows: list[dict], label_suffix: str) -> tuple[float | None, int, float]:
    """Observed seconds/item for every vLLM-backed ledger row whose label ends in
    `label_suffix`, plus the row count and total seconds it was computed from -- so a caller can
    say how much history a rate rests on rather than presenting a bare number."""
    total_s = 0.0
    total_items = 0
    n_rows = 0
    for row in ledger_rows:
        if not row.get("label", "").endswith(label_suffix):
            continue
        if row.get("backend") != "vllm":
            continue
        items = int(row.get("items") or 0)
        seconds = float(row.get("gpu_seconds") or 0)
        if items <= 0:
            continue
        total_s += seconds
        total_items += items
        n_rows += 1
    rate = (total_s / total_items) if total_items else None
    return rate, n_rows, total_s


# E3's own condition count (N0/N1/N2). Kept as a local literal rather than importing
# `run_necessity.CONDITION_LABELS`: that module also imports `pilot.models`, which this section
# has no other reason to pull in, and the count is a documented, stable fact of the design this
# task also states directly (three conditions, N0-N2).
E3_CONDITIONS = 3


def project_e1_cost(
    ledger_rows: list[dict], *, cumulative_gpu_s: float,
    reask_rate_s: float | None = None, probe_rate_s: float | None = None,
    load_overhead_s: float = ASSUMED_MODEL_LOAD_S,
    built_low: int = E1_BUILT_TARGET_LOW, built_high: int = E1_BUILT_TARGET_HIGH,
    yield_rate: float = E1_BUILT_YIELD_4OPT, checkpoints: int = E1_CHECKPOINTS,
    conditions: int = E1_CONDITIONS, budget_s: float = C.PROJECT_GPU_BUDGET_S,
) -> dict:
    """GPU-seconds/hours for E1 (a fourth model lineage, `checkpoints` checkpoints, 4 options,
    `built_low`-`built_high` built items) from observed per-item rates.

    Per checkpoint: stage 1 (the free-text elicitation that selects the attempted pool) costs
    `attempted_items * reask_rate_s`. Stage 3 costs `built_items * conditions * per_condition_s`
    -- one free-text re-ask plus one forced-choice probe per condition, the shape
    `run_experiment.run_stage3`/`run_necessity.run_stage_necessity` both use. `per_condition_s`
    is read, in preference order, off:

      1. committed `/necessity` ledger rows (round 6's own E3 run, once it exists) -- the same
         two-call-per-condition shape as E1's own stage 3, just over `E3_CONDITIONS` (3) instead
         of `conditions` (5 for E1's R0-R4), so its per-item rate divided by 3 and scaled back up
         by `conditions` is the best-grounded estimate this function can make;
      2. `reask_rate_s + probe_rate_s` (from `/stage1` and `/probe_variants` ledger rows) --
         E1-shaped calls from two different experiments, summed as a stand-in.

    `attempted_items` for a target built-item count is `ceil(built / yield_rate)`; low/high are
    computed independently (not resampled) since this is a point projection over a handful of
    observed rates, not a statistic with its own sampling distribution.
    """
    if reask_rate_s is None:
        reask_rate_s, _n, _s = _rate_from_ledger(ledger_rows, "/stage1")
    if probe_rate_s is None:
        probe_rate_s, _n, _s = _rate_from_ledger(ledger_rows, "/probe_variants")
    necessity_rate_s, necessity_n, _s = _rate_from_ledger(ledger_rows, "/necessity")

    assumptions: list[str] = []
    if reask_rate_s is None:
        reask_rate_s = 2.2  # ponytail: mean of this project's own historical /stage1 rows
        assumptions.append(
            f"no /stage1 ledger rows found; assumed re-ask rate {reask_rate_s:.2f} s/item "
            "(this project's own historical average)"
        )
    if probe_rate_s is None:
        probe_rate_s = 0.15  # ponytail: mean of this project's own historical /probe_variants rows
        assumptions.append(
            f"no /probe_variants ledger rows found; assumed forced-probe rate "
            f"{probe_rate_s:.2f} s/row (this project's own historical average)"
        )
    assumptions.append(
        f"model-load overhead is not recoverable from the ledger (see ASSUMED_MODEL_LOAD_S); "
        f"assumed {load_overhead_s:.0f}s/checkpoint load"
    )

    if necessity_rate_s is not None:
        per_condition_s = necessity_rate_s / E3_CONDITIONS
        stage3_source = (
            f"{necessity_n} committed /necessity ledger row(s): {necessity_rate_s:.3f} s/item "
            f"over {E3_CONDITIONS} conditions = {per_condition_s:.3f} s/condition"
        )
    else:
        per_condition_s = reask_rate_s + probe_rate_s
        stage3_source = (
            f"/stage1 + /probe_variants sums ({reask_rate_s:.2f} + {probe_rate_s:.2f} "
            f"= {per_condition_s:.2f} s/condition) -- no /necessity ledger rows yet, so stage "
            "3's own free-text re-ask is assumed to cost the same per item as stage 1's"
        )
        assumptions.append(
            "no /necessity ledger rows found (E3 has not run yet); stage 3's per-condition cost "
            f"is assumed to be /stage1's re-ask rate plus /probe_variants' forced-probe rate "
            f"({per_condition_s:.2f} s/condition) rather than measured directly"
        )

    def _per_checkpoint(built: int) -> dict:
        import math
        attempted = math.ceil(built / yield_rate)
        stage1_s = attempted * reask_rate_s
        stage3_s = built * conditions * per_condition_s
        total = stage1_s + stage3_s + load_overhead_s
        return {"built": built, "attempted": attempted, "stage1_s": stage1_s,
                "stage3_s": stage3_s, "load_s": load_overhead_s, "total_s": total}

    low = _per_checkpoint(built_low)
    high = _per_checkpoint(built_high)
    total_low_s = low["total_s"] * checkpoints
    total_high_s = high["total_s"] * checkpoints
    remaining_s = budget_s - cumulative_gpu_s

    return {
        "reask_rate_s_per_item": reask_rate_s, "probe_rate_s_per_row": probe_rate_s,
        "necessity_rate_s_per_item": necessity_rate_s, "per_condition_s": per_condition_s,
        "stage3_source": stage3_source,
        "load_overhead_s_per_checkpoint": load_overhead_s, "checkpoints": checkpoints,
        "conditions_per_item": conditions, "yield_rate": yield_rate,
        "per_checkpoint_low": low, "per_checkpoint_high": high,
        "total_low_s": total_low_s, "total_high_s": total_high_s,
        "total_low_h": total_low_s / 3600, "total_high_h": total_high_s / 3600,
        "cumulative_gpu_s": cumulative_gpu_s, "budget_s": budget_s,
        "remaining_s": remaining_s, "remaining_h": remaining_s / 3600,
        "fits_low": total_low_s <= remaining_s, "fits_high": total_high_s <= remaining_s,
        "assumptions": assumptions,
    }


def report_e1_estimate(results_dir: Path) -> tuple[str, dict]:
    from .watchdog import cumulative_gpu_seconds

    lines: list[str] = []
    w = lines.append
    w("## 8. E1 cost projection")
    w("")
    w("Projected GPU cost of E1 against a fourth, independent model lineage: 3 checkpoints,")
    w("4 options, targeting 300-390 built items at the measured 32.1% built-item yield for four")
    w("options (Part A: 387/1204 attempted, already published and audited for Part A).")
    w("")

    ledger_rows = load_ledger_rows(C.GPU_LEDGER)
    cumulative = cumulative_gpu_seconds(C.GPU_LEDGER)

    found_round6 = [f for f in ROUND6_SUMMARY_FILES if (results_dir / f).exists()]
    if found_round6:
        w(f"Round-6 summary file(s) present under `--results-dir`: {', '.join(found_round6)}.")
    else:
        w("No round-6 run summary present yet under `--results-dir` (this section is being run "
          "before the pod work is done).")
    w("")

    proj = project_e1_cost(ledger_rows, cumulative_gpu_s=cumulative)

    w(f"Re-ask rate: **{proj['reask_rate_s_per_item']:.3f} s/item** "
      f"(from committed `/stage1` ledger rows). Forced-probe rate: "
      f"**{proj['probe_rate_s_per_row']:.3f} s/row** (from committed `/probe_variants` ledger "
      f"rows). Load overhead: **{proj['load_overhead_s_per_checkpoint']:.0f} s/checkpoint** "
      "(assumed, not measured -- see below).")
    w("")
    w(f"Stage 3's per-condition cost (one free-text re-ask plus one forced-choice probe): "
      f"**{proj['per_condition_s']:.3f} s/condition**, from {proj['stage3_source']}.")
    w("")
    w("| bound | built items | attempted items | stage 1 (h) | stage 3 (h) | load (h) "
      "| total/checkpoint (h) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for label, d in (("low", proj["per_checkpoint_low"]), ("high", proj["per_checkpoint_high"])):
        w(f"| {label} | {d['built']} | {d['attempted']} | {d['stage1_s'] / 3600:.2f} "
          f"| {d['stage3_s'] / 3600:.2f} | {d['load_s'] / 3600:.3f} "
          f"| {d['total_s'] / 3600:.2f} |")
    w("")
    w(f"**Total for 3 checkpoints: {proj['total_low_h']:.2f}-{proj['total_high_h']:.2f} GPU-hours "
      f"({proj['total_low_s']:.0f}-{proj['total_high_s']:.0f} s).** Wall-clock is the same figure")
    w("(this project runs one model at a time on one GPU, so GPU-seconds and wall-clock seconds")
    w("coincide here).")
    w("")
    w(f"Cumulative GPU-seconds already spent (from `{C.GPU_LEDGER.name}`): "
      f"{proj['cumulative_gpu_s']:.1f} s ({proj['cumulative_gpu_s'] / 3600:.2f} h). Project "
      f"budget: {proj['budget_s']:.0f} s ({proj['budget_s'] / 3600:.1f} h). Remaining: "
      f"{proj['remaining_s']:.1f} s ({proj['remaining_h']:.2f} h).")
    w("")
    fits = "fits" if proj["fits_high"] else ("fits at the low end only" if proj["fits_low"] else "does not fit")
    w(f"**E1 {fits} the remaining budget.**")
    w("")
    w("Assumptions this projection depends on:")
    w("")
    for a in proj["assumptions"]:
        w(f"- {a}")
    w("")
    return "\n".join(lines) + "\n", proj


# ============================================================================== reporting


SECTIONS = {
    "integrity": lambda nested, results_dir: report_integrity(nested),
    "discrete": lambda nested, results_dir: report_discrete(nested),
    "continuous": lambda nested, results_dir: report_continuous(nested),
    "loo": lambda nested, results_dir: report_loo(nested),
    "heterogeneity": lambda nested, results_dir: report_heterogeneity(nested),
    "strata": lambda nested, results_dir: report_strata(nested),
    "power": lambda nested, results_dir: report_power(nested),
    "e1-estimate": lambda nested, results_dir: report_e1_estimate(results_dir),
}


def build_report(results_dir: Path, sections: list[str]) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    w("# E3 (necessity) offline analysis")
    w("")
    w("Generated by `python -m pilot.analyze_necessity` from committed `stage_necessity_*.jsonl`")
    w(f"under `{results_dir}`. No GPU, no network, no generation. Every bootstrap uses seed")
    w(f"`{SEED}` with `{RESAMPLES}` resamples, matching `pilot/analyze_run3.py` and")
    w("`pilot/analyze_round5.py`; every exact test is the same `math.comb`-based binomial")
    w("`run_experiment.mcnemar` uses.")
    w("")

    rows = load_necessity_rows(results_dir)
    nested = build_nested(rows)
    n_models = len(nested)
    n_items = sum(len(v) for v in nested.values())
    if not rows:
        w("**No `stage_necessity_*.jsonl` found under this results directory yet.** Every")
        w("section below still runs (on an empty population) so this report is exercisable")
        w("before the pod work is done; see `--self-check` for arithmetic verified against a")
        w("built-in fixture instead of committed files.")
    else:
        w(f"{len(rows)} row(s), {n_models} model(s), {n_items} item(s) loaded.")
    w("")

    payload: dict = {
        "seed": SEED, "n_resamples": RESAMPLES, "sections": sections,
        "n_rows": len(rows), "n_models": n_models, "n_items": n_items, "results": {},
    }
    for section in sections:
        text, section_payload = SECTIONS[section](nested, results_dir)
        lines.append(text)
        payload["results"][section] = section_payload
    return "\n".join(lines), payload


# ============================================================================== self-check


def _fixture_rows() -> list[dict]:
    """A small, hand-built E3 result set covering: two models, an N0 flip, gold vs non-gold and
    relevant vs irrelevant items, and a missing `delta_p_original` (the probe not completing) --
    every branch every section above has to handle. Numbers are chosen so b/c and the mean
    delta p are checkable by hand, not just internally consistent.

    `mcnemar(outcomes, "N1", "N2")` counts `b_a_only` as N1 True / N2 False (control-only flip --
    the direction opposite the necessity hypothesis) and `c_b_only` as N1 False / N2 True (the
    informative direction: deleting the named attribute flipped the choice, the length-matched
    control did not).

    Model M1: 6 items, all N0-reproducing.
      i1, i2: N1=False, N2=True   -> discordant, informative direction (c)
      i3:     N1=False, N2=False  -> concordant
      i4:     N1=True,  N2=False  -> discordant, control-only direction (b)
      i5, i6: N1=True,  N2=True   -> concordant
      -> N1 vs N2 on M1: b_a_only=1 (i4), c_b_only=2 (i1, i2).
    Model M2: 3 items, one N0 flip (excluded), two N0-reproducing.
      i7: N0=False (flip) -- excluded entirely from every contrast.
      i8: N1=False, N2=True  -> discordant, informative direction (c)
      i9: N1=False, N2=True  -> discordant, informative direction (c)
      -> N1 vs N2 on M2 (interpretable only): b_a_only=0, c_b_only=2.
    Pooled N1 vs N2: b_a_only=1, c_b_only=4.
    """
    rows: list[dict] = []

    def add(model, item_id, condition, *, n0=True, n1=None, n2=None, gold=True, relevant=False,
            delta=None):
        still = {"N0": n0, "N1": n1, "N2": n2}[condition]
        rows.append({
            "model": model, "part": "A", "item_id": item_id, "condition": condition,
            "attribute": "director", "rival_letter": "B", "rival_title": "Rival",
            "chosen_letter": "A", "chosen_is_gold": gold,
            "question_relevant_attribute": relevant, "choice": "A" if still else "B",
            "still_chooses_original": still, "letter_probe": {}, "p_original": None,
            "r0_p_target": None, "delta_p_original": delta, "response": "stub",
        })

    # M1
    add("M1", "i1", "N0", n0=True, gold=True, relevant=False)
    add("M1", "i1", "N1", n1=False, gold=True, relevant=False, delta=-0.30)
    add("M1", "i1", "N2", n2=True, gold=True, relevant=False, delta=-0.05)
    add("M1", "i2", "N0", n0=True, gold=False, relevant=True)
    add("M1", "i2", "N1", n1=False, gold=False, relevant=True, delta=-0.40)
    add("M1", "i2", "N2", n2=True, gold=False, relevant=True, delta=0.02)
    add("M1", "i3", "N0", n0=True, gold=True, relevant=False)
    add("M1", "i3", "N1", n1=False, gold=True, relevant=False, delta=-0.10)
    add("M1", "i3", "N2", n2=False, gold=True, relevant=False, delta=-0.12)
    add("M1", "i4", "N0", n0=True, gold=True, relevant=False)
    add("M1", "i4", "N1", n1=True, gold=True, relevant=False, delta=0.01)
    add("M1", "i4", "N2", n2=False, gold=True, relevant=False, delta=-0.20)
    add("M1", "i5", "N0", n0=True, gold=False, relevant=False)
    add("M1", "i5", "N1", n1=True, gold=False, relevant=False, delta=None)  # probe incomplete
    add("M1", "i5", "N2", n2=True, gold=False, relevant=False, delta=0.03)
    add("M1", "i6", "N0", n0=True, gold=True, relevant=False)
    add("M1", "i6", "N1", n1=True, gold=True, relevant=False, delta=0.02)
    add("M1", "i6", "N2", n2=True, gold=True, relevant=False, delta=0.01)

    # M2
    add("M2", "i7", "N0", n0=False, gold=True, relevant=False)  # flipped -- excluded
    add("M2", "i7", "N1", n1=True, gold=True, relevant=False, delta=0.5)
    add("M2", "i7", "N2", n2=True, gold=True, relevant=False, delta=0.5)
    add("M2", "i8", "N0", n0=True, gold=True, relevant=True)
    add("M2", "i8", "N1", n1=False, gold=True, relevant=True, delta=-0.25)
    add("M2", "i8", "N2", n2=True, gold=True, relevant=True, delta=0.00)
    add("M2", "i9", "N0", n0=True, gold=False, relevant=False)
    add("M2", "i9", "N1", n1=False, gold=False, relevant=False, delta=-0.35)
    add("M2", "i9", "N2", n2=True, gold=False, relevant=False, delta=-0.01)

    return rows


def self_check() -> int:
    """Every section's arithmetic against the hand-checkable fixture above, plus the E1
    projection against a synthetic ledger -- no files needed for either."""
    problems: list[str] = []

    rows = _fixture_rows()
    nested = build_nested(rows)

    # -- integrity
    _, integrity = report_integrity(nested)
    if integrity["pooled"]["flipped"] != 1:
        problems.append(f"integrity: expected 1 pooled N0 flip, got {integrity['pooled']}")
    if integrity["per_model"]["M2"]["flipped"] != 1:
        problems.append("integrity: expected M2's flip to be attributed to M2")

    # -- discrete: pooled N1 vs N2 should be b=4 (M1:2, M2:2), c=1 (M1:1, M2:0)
    _, discrete = report_discrete(nested)
    pooled_n1n2 = discrete["N1-N2"]["**pooled**"]
    if (pooled_n1n2["b_a_only"], pooled_n1n2["c_b_only"]) != (1, 4):
        problems.append(
            f"discrete N1-N2 pooled: expected b=1,c=4, got "
            f"b={pooled_n1n2['b_a_only']},c={pooled_n1n2['c_b_only']}"
        )
    m1_n1n2 = discrete["N1-N2"]["M1"]
    if (m1_n1n2["b_a_only"], m1_n1n2["c_b_only"]) != (1, 2):
        problems.append(
            f"discrete N1-N2 M1: expected b=1,c=2, got "
            f"b={m1_n1n2['b_a_only']},c={m1_n1n2['c_b_only']}"
        )

    # -- continuous: M1's N1-N2 delta_p_original diffs, over items complete on both sides
    # (i5 excluded: N1's delta is None there). i1: -0.30-(-0.05)=-0.25; i2: -0.40-0.02=-0.42;
    # i3: -0.10-(-0.12)=0.02; i4: 0.01-(-0.20)=0.21; i6: 0.02-0.01=0.01. n=5.
    _, continuous = report_continuous(nested)
    m1_cont = continuous["N1-N2"]["M1"]
    if m1_cont["n"] != 5:
        problems.append(f"continuous N1-N2 M1: expected n=5 (i5 excluded), got {m1_cont['n']}")
    hand_mean = (-0.25 - 0.42 + 0.02 + 0.21 + 0.01) / 5
    if m1_cont["mean"] is None or abs(m1_cont["mean"] - hand_mean) > 1e-9:
        problems.append(f"continuous N1-N2 M1: expected mean {hand_mean}, got {m1_cont['mean']}")

    # -- strata: gold vs not_gold, N1-N2, must route i1/i3/i4/i6/i8 to "gold" and i2/i5/i9 to
    # "not_gold" (i7 excluded as an N0 flip regardless of its own gold flag)
    _, strata = report_strata(nested)
    gold_strat = strata["chosen_is_gold"]["N1-N2"]
    if gold_strat["gold"]["n_rows"] != 5:
        problems.append(f"strata chosen_is_gold: expected gold n_rows=5, got {gold_strat['gold']['n_rows']}")
    if gold_strat["not_gold"]["n_rows"] != 3:
        problems.append(
            f"strata chosen_is_gold: expected not_gold n_rows=3, got "
            f"{gold_strat['not_gold']['n_rows']}"
        )
    rel_strat = strata["question_relevant_attribute"]["N1-N2"]
    if rel_strat["relevant"]["n_rows"] != 2 or rel_strat["not_relevant"]["n_rows"] != 6:
        problems.append(f"strata question_relevant_attribute: unexpected bucket sizes {rel_strat}")

    # -- power: a discordant rate of exactly 0 must not raise, and must report n/a not a crash
    zero_nested = build_nested([r for r in rows if r["model"] == "M1" and r["item_id"] == "i6"])
    zero_nested["M1"]["i6"]["N1"]["still_chooses_original"] = True
    zero_nested["M1"]["i6"]["N2"]["still_chooses_original"] = True
    try:
        report_power(zero_nested)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"power section raised on a zero-discordant fixture: {exc}")

    # -- loo / heterogeneity: must not raise on this fixture, and dropping the only-flip model
    # must not change M1's own N1-N2 b/c (each model resamples nothing, so this is exact)
    loo_text, loo_payload = report_loo(nested)
    dropped_m2 = next(r for r in loo_payload["N1-N2"] if r["dropped"] == "M2")
    if (dropped_m2["discrete"]["b_a_only"], dropped_m2["discrete"]["c_b_only"]) != (1, 2):
        problems.append(f"loo: dropping M2 should leave M1's own b=1,c=2, got {dropped_m2['discrete']}")
    report_heterogeneity(nested)  # must not raise

    # -- e1-estimate: hand-computable synthetic ledger and cumulative total
    synthetic_ledger = [
        {"label": "X/stage1", "backend": "vllm", "items": "100", "gpu_seconds": "200.0"},
        {"label": "Y/stage1", "backend": "vllm", "items": "100", "gpu_seconds": "300.0"},
        {"label": "X/probe_variants", "backend": "vllm", "items": "1000", "gpu_seconds": "100.0"},
        # a non-vllm and a zero-item row must both be ignored, not divide by zero or skew the mean
        {"label": "Z/stage1", "backend": "transformers", "items": "50", "gpu_seconds": "999.0"},
        {"label": "W/stage1", "backend": "vllm", "items": "0", "gpu_seconds": "5.0"},
    ]
    proj = project_e1_cost(synthetic_ledger, cumulative_gpu_s=1000.0, budget_s=100_000.0)
    # reask rate = (200+300)/(100+100) = 2.5 s/item; probe rate = 100/1000 = 0.1 s/row
    if abs(proj["reask_rate_s_per_item"] - 2.5) > 1e-9:
        problems.append(f"e1-estimate: expected reask rate 2.5, got {proj['reask_rate_s_per_item']}")
    if abs(proj["probe_rate_s_per_row"] - 0.1) > 1e-9:
        problems.append(f"e1-estimate: expected probe rate 0.1, got {proj['probe_rate_s_per_row']}")
    # hand-check the low bound: built=300, yield=387/1204 -> attempted=ceil(300/(387/1204))=934
    import math
    expect_attempted_low = math.ceil(300 / (387 / 1204))
    if proj["per_checkpoint_low"]["attempted"] != expect_attempted_low:
        problems.append(
            f"e1-estimate: expected attempted(low)={expect_attempted_low}, got "
            f"{proj['per_checkpoint_low']['attempted']}"
        )
    stage1_low = expect_attempted_low * 2.5
    stage3_low = 300 * 5 * (2.5 + 0.1)
    expect_total_low = (stage1_low + stage3_low + ASSUMED_MODEL_LOAD_S) * 3
    if abs(proj["total_low_s"] - expect_total_low) > 1e-6:
        problems.append(f"e1-estimate: expected total_low_s={expect_total_low}, got {proj['total_low_s']}")
    if proj["remaining_s"] != 100_000.0 - 1000.0:
        problems.append(f"e1-estimate: remaining_s wrong: {proj['remaining_s']}")

    # -- e1-estimate fallback: an empty ledger must still return a usable, assumption-flagged
    # projection rather than raising or dividing by zero
    fallback = project_e1_cost([], cumulative_gpu_s=0.0)
    if not fallback["assumptions"]:
        problems.append("e1-estimate: empty-ledger fallback produced no stated assumptions")
    if fallback["reask_rate_s_per_item"] is None or fallback["probe_rate_s_per_row"] is None:
        problems.append("e1-estimate: empty-ledger fallback left a rate as None")

    # -- e1-estimate: once /necessity ledger rows exist (E3 has actually run), stage 3's
    # per-condition cost must be read from them -- 3.0 s/item over 3 conditions = 1.0 s/condition
    # -- rather than falling back to the /stage1+/probe_variants sum.
    ledger_with_necessity = synthetic_ledger + [
        {"label": "X/necessity", "backend": "vllm", "items": "50", "gpu_seconds": "150.0"},
    ]
    proj_n = project_e1_cost(ledger_with_necessity, cumulative_gpu_s=1000.0, budget_s=100_000.0)
    if abs(proj_n["per_condition_s"] - 1.0) > 1e-9:
        problems.append(
            f"e1-estimate: expected /necessity-derived per_condition_s=1.0, got "
            f"{proj_n['per_condition_s']}"
        )
    stage3_low_n = 300 * 5 * 1.0
    expect_total_low_n = (expect_attempted_low * 2.5 + stage3_low_n + ASSUMED_MODEL_LOAD_S) * 3
    if abs(proj_n["total_low_s"] - expect_total_low_n) > 1e-6:
        problems.append(
            f"e1-estimate: /necessity-preferred total_low_s expected {expect_total_low_n}, got "
            f"{proj_n['total_low_s']}"
        )

    # -- ledger header verification: a header missing a required column must raise
    try:
        _verify_ledger_header(["timestamp_utc", "label"])
        problems.append("_verify_ledger_header: should have raised on a truncated header")
    except ValueError:
        pass
    _verify_ledger_header(list(LEDGER_FIELDS))  # must not raise on the real header

    for p in problems:
        print("FAIL", p)
    print("self-check:", "ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


# ============================================================================== CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR,
                     help=f"where to look for stage_necessity_*.jsonl (default: {C.RESULTS_DIR})")
    ap.add_argument("--out-dir", type=Path, default=C.REPO_ROOT / "code",
                     help="where to write the report and its JSON sidecar (default: code/, "
                          "never code/results/ -- committed run records are read-only)")
    ap.add_argument("--section", choices=[*SECTIONS, "all"], default="all")
    ap.add_argument("--self-check", action="store_true", help="arithmetic checks only, no files")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()

    sections = list(SECTIONS) if args.section == "all" else [args.section]
    report, payload = build_report(args.results_dir, sections)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "analysis_necessity_report.md"
    json_path = args.out_dir / "analysis_necessity_report.json"
    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
