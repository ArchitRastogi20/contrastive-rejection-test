"""Offline analyses added in round 5, computed from the committed stage-2/stage-3 JSONL alone.

No new generation for sections 1-4: every number there is a deterministic function of files
already in ``code/results/``. Section 5 additionally rebuilds the item corpus offline (the same
2WikiMultihopQA dump the run itself used, already cached locally) to recover which candidate the
repair search actually accepted, by indexing the project's own ordered candidate lists with the
1-based positions the stage-2 records already store -- it never replays the search itself.
Section 6 reads the per-item surprisal ``pilot/surprisal.py`` already measured and committed
(``code/results/surprisal___*.jsonl``) and joins it to the stage-3 choice; it does not call a
model or recompute a surprisal reading either.

    python -m pilot.analyze_round5                          # everything, all three parts
    python -m pilot.analyze_round5 --part A --section loo    # one part, one section
    python -m pilot.analyze_round5 --section provenance      # section 5 alone, needs the dataset
    python -m pilot.analyze_round5 --section fluency         # section 6 alone, Part A + Part C

Six families of numbers, none of which ``pilot/analyze_run3.py`` produces:

    1. leave-one-model-out, both measures, three contrasts, each part
    2. Cochran's Q / I^2 between-model heterogeneity over the same three contrasts
    3. item-clustered (rather than row-clustered) percentile bootstrap, as a check on pooling
    4. per-relation stratification of the same three contrasts, small relations pooled as "other"
    5. where the R1/R3 (relevant) and R2/R4 (control) edit sentences actually come from
    6. whether the R1-R2/R3-R4 content contrasts survive conditioning on ``pilot/surprisal.py``'s
       own per-item fluency gap (``--section fluency``) -- the tercile/leave-one-out analysis
       ``pilot/surprisal.py`` itself only measures the raw gap for, never conditions on
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Iterable
from pathlib import Path

from . import config as C
from . import repair
from .analyze_run3 import (
    PARTS,
    _rows,
    discrete_contrast,
    discrete_outcomes,
    load_part,
    paired_mean,
    short_model,
)
from .data import build_items, load_records
from .extract import Rejection
from .run_experiment import bootstrap_ci_mean_diff, mcnemar  # noqa: F401 -- reused via analyze_run3

# Same discipline as analyze_run3: one seed, recorded once, used by every bootstrap below.
SEED = 20260822
RESAMPLES = 10000

# The three contrasts round 5 asks about: the two content contrasts (R3-R4 at the
# un-complained-about third option, R1-R2 at the rival itself) and the one location/content
# interaction contrast (R2-R4) the paper's measurement-disagreement discussion turns on.
ROUND5_CONTRASTS = [("R3", "R4"), ("R1", "R2"), ("R2", "R4")]


# ============================================================================== shared pooling


def _pooled_discrete(models: dict, exclude: frozenset = frozenset()) -> dict:
    """``chosen_is_edited`` per item, pooled over every model in ``models`` except ``exclude``,
    keyed ``"{model}::{item_id}"`` -- the same key scheme ``analyze_run3.build_report`` pools
    with, so a dropped model truly disappears from the paired population rather than merely
    being excluded from a display column."""
    pooled = {
        f"{m}::{i}": conds
        for m in sorted(models) if m not in exclude
        for i, conds in models[m].items()
    }
    return discrete_outcomes(pooled)


def _pooled_continuous(models: dict, exclude: frozenset = frozenset()) -> dict:
    """The ``delta_p_edited`` counterpart of ``_pooled_discrete``, in the same key scheme."""
    out: dict[str, dict] = {}
    for m in sorted(models):
        if m in exclude:
            continue
        for item_id, conds in models[m].items():
            out[f"{m}::{item_id}"] = {
                c: {"delta_p_edited": row.get("delta_p_edited")} for c, row in conds.items()
            }
    return out


# ==================================================================== 1. leave-one-model-out


def leave_one_model_out(models: dict, a: str, b: str) -> list[dict]:
    """Discrete + continuous ``a-b`` contrast pooled over every model, then again with each
    model dropped in turn -- the population regression targets are checked against."""
    rows = []
    disc_all = _pooled_discrete(models)
    cont_all = _pooled_continuous(models)
    rows.append({
        "dropped": None,
        "discrete": discrete_contrast(disc_all, a, b),
        "continuous": paired_mean(cont_all, a, b, "delta_p_edited", seed=SEED),
    })
    for m in sorted(models):
        disc = _pooled_discrete(models, exclude={m})
        cont = _pooled_continuous(models, exclude={m})
        rows.append({
            "dropped": short_model(m),
            "discrete": discrete_contrast(disc, a, b),
            "continuous": paired_mean(cont, a, b, "delta_p_edited", seed=SEED),
        })
    return rows


# ============================================================ 2. between-model heterogeneity


def _log_or(b: int, c: int) -> tuple[float, float]:
    """Log odds ratio and its Wald variance for one model's discordant pair counts, with the
    same Haldane-Anscombe +0.5 correction ``analyze_run3.odds_ratio`` applies (and only under
    the same condition: a discordant cell of exactly 0), so a per-model estimate here is
    comparable to that function's own pooled one rather than merely similarly shaped."""
    if b == 0 or c == 0:
        bb, cc = b + 0.5, c + 0.5
    else:
        bb, cc = float(b), float(c)
    return math.log(bb / cc), 1.0 / bb + 1.0 / cc


def _chi2_sf_even_df(x: float, df: int) -> float:
    """Exact chi-squared survival function for an even ``df``, no scipy needed: a chi-squared
    variable with ``2m`` degrees of freedom is the sum of ``m`` iid Exponential(rate=1/2)
    variables, which gives its survival function a short closed-form finite sum (the Erlang/
    Poisson-process tail identity). Every heterogeneity test in this module has
    ``df = n_models - 1 = 2`` (every roster this project uses has exactly three models); this is
    written for general even ``df`` rather than hardcoded to 2 only because the general form
    costs nothing extra and the derivation is the same either way.
    """
    if df <= 0 or df % 2 != 0:
        raise ValueError(f"even positive df required, got {df}")
    if x <= 0:
        return 1.0
    m = df // 2
    half_x = x / 2.0
    total = 0.0
    term = 1.0  # (half_x ** 0) / 0!
    for i in range(m):
        if i > 0:
            term *= half_x / i
        total += term
    return math.exp(-half_x) * total


def heterogeneity(models: dict, a: str, b: str) -> dict:
    """Cochran's Q, its degrees of freedom, an exact p-value, and I^2 over the per-model log
    odds ratios for one contrast, each model's own (unpooled) discordant pair counts.

    tau^2 (the between-study variance DerSimonian-Laird would estimate) is deliberately not
    computed or reported anywhere in this module: with three models -- two degrees of freedom --
    it would be estimated from almost no information, and quoting a number for it would dress up
    a spread Q itself cannot distinguish from sampling noise.
    """
    logors: list[float] = []
    weights: list[float] = []
    per_model: list[dict] = []
    for m in sorted(models):
        d = discrete_contrast(discrete_outcomes(models[m]), a, b)
        b_, c_ = d["b_a_only"], d["c_b_only"]
        logor, var = _log_or(b_, c_)
        logors.append(logor)
        weights.append(1.0 / var)
        per_model.append({"model": short_model(m), "b": b_, "c": c_,
                           "or": math.exp(logor), "log_or": logor, "var": var})
    total_weight = sum(weights)
    if not weights or total_weight == 0:
        return {"q": None, "df": None, "p": None, "i2": None, "pooled_log_or": None,
                "per_model": per_model}
    pooled_log_or = sum(w * l for w, l in zip(weights, logors)) / total_weight
    q = sum(w * (l - pooled_log_or) ** 2 for w, l in zip(weights, logors))
    df = len(logors) - 1
    p = _chi2_sf_even_df(q, df) if df > 0 and df % 2 == 0 else None
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return {"q": q, "df": df, "p": p, "i2": i2, "pooled_log_or": pooled_log_or,
            "per_model": per_model}


# ============================================================== 3. item-clustered bootstrap


def _cluster_diffs_continuous(models: dict, a: str, b: str) -> dict[str, list[float]]:
    """``{item_id: [delta_p_edited diffs, one per model with a complete pair]}`` -- the real
    dataset item id, not the ``"model::item_id"`` pooling key, so an item that several models
    both built and completed contributes several diffs to the *same* cluster.

    Built walking ``models`` in sorted order and each model's own item dict in its own
    (insertion) order, never re-sorted by item id -- this is the order the seeded resample below
    draws cluster positions from, so changing it would silently change every reported interval
    even though the point estimate (a plain sum over all diffs either way) would not move."""
    clusters: dict[str, list[float]] = {}
    for m in sorted(models):
        for item_id, conds in models[m].items():
            ra, rb = conds.get(a), conds.get(b)
            if ra is None or rb is None:
                continue
            va, vb = ra.get("delta_p_edited"), rb.get("delta_p_edited")
            if va is None or vb is None:
                continue
            clusters.setdefault(item_id, []).append(va - vb)
    return clusters


def _cluster_diffs_discrete(models: dict, a: str, b: str) -> dict[str, list[float]]:
    """The 0/1 (``chosen_is_edited``) paired-risk-difference counterpart of
    ``_cluster_diffs_continuous``, same clustering and ordering rules."""
    clusters: dict[str, list[float]] = {}
    for m in sorted(models):
        for item_id, conds in models[m].items():
            ra, rb = conds.get(a), conds.get(b)
            if ra is None or rb is None:
                continue
            va, vb = ra.get("chosen_is_edited"), rb.get("chosen_is_edited")
            if va is None or vb is None:
                continue
            clusters.setdefault(item_id, []).append(float(bool(va)) - float(bool(vb)))
    return clusters


def cluster_bootstrap_mean_diff(
    clusters: dict[str, list[float]], *, seed: int = SEED, n_resamples: int = RESAMPLES,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for the mean of every diff in ``clusters``, resampling whole
    clusters (real item ids) with replacement rather than rows.

    An item that contributed several models' worth of paired rows moves as one unit: either all
    of its rows are in a given resample (possibly more than once, since this is sampling with
    replacement) or none are, so it cannot be drawn into a resample more times than a
    single-model item just because it happened to survive to more models. Same resampling
    discipline as ``run_experiment.bootstrap_ci_mean_diff`` otherwise: one seeded
    ``random.Random``, ``n_resamples`` replicates each drawing exactly as many clusters as exist
    in ``clusters``, same percentile index formula.
    """
    cluster_ids = list(clusters)
    k = len(cluster_ids)
    all_diffs = [d for cid in cluster_ids for d in clusters[cid]]
    n = len(all_diffs)
    if k == 0 or n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_clusters": 0,
                "seed": seed, "n_resamples": n_resamples}
    mean = sum(all_diffs) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        resample: list[float] = []
        for _ in range(k):
            resample.extend(clusters[cluster_ids[rng.randrange(k)]])
        means.append(sum(resample) / len(resample) if resample else 0.0)
    means.sort()
    lo_idx = min(n_resamples - 1, max(0, int((alpha / 2) * n_resamples)))
    hi_idx = min(n_resamples - 1, max(0, int((1 - alpha / 2) * n_resamples) - 1))
    return {"mean": mean, "ci_low": means[lo_idx], "ci_high": means[hi_idx], "n": n,
            "n_clusters": k, "seed": seed, "n_resamples": n_resamples}


def distinct_built_item_ids(results: Path, part: str) -> int:
    """How many distinct dataset item ids have at least one ``built: true`` stage-2 row in this
    part, over every model -- the size of the item universe the part's built rows are drawn
    from, reported alongside the item-clustered intervals as context for how much row/cluster
    overlap the pooled bootstrap in section 1 of ``analyze_run3.py`` is exposed to."""
    directory, _ = PARTS[part]
    ids: set[str] = set()
    for path in sorted((results / directory).glob("stage2_*.jsonl")):
        for row in _rows(path):
            if row.get("built"):
                ids.add(row["item_id"])
    return len(ids)


# ========================================================= 4. per-relation stratification


# Below this many (model, item) pairs, a relation's own stratum would be too small to read a
# contrast off on its own, so it is folded into "other" instead. This corpus's attribute
# vocabulary supports up to a dozen relations (see extract.ATTRIBUTES); in every part built here
# only two of them -- the age/identity fact the model tends to name (date_of_birth) and the
# creative-work fact the roster leans on (director) -- clear a triple-digit item count on their
# own, so 100 separates "large enough to read alone" from "everything else", not a threshold
# tuned to include or exclude any particular relation.
MIN_ROWS_PER_RELATION = 100


def item_attribute(conds: dict) -> str | None:
    """The named attribute for one item -- a property of the rejection (hence of the
    model,item pair), read off whichever condition row happens to carry it, since every
    condition of a given built item shares the same ``attribute`` field."""
    for row in conds.values():
        if row.get("attribute"):
            return row["attribute"]
    return None


def relation_buckets(models: dict, min_rows: int = MIN_ROWS_PER_RELATION) -> dict[str, list[str]]:
    """``{bucket_name: [pooled "model::item_id" keys]}`` -- every relation at or above
    ``min_rows`` (model, item) pairs gets its own bucket; everything else is pooled into
    ``"other"``.

    Keys are kept in the order first encountered walking ``models`` in sorted order and each
    model's own item dict in its own (insertion) order -- never collected through a ``set`` or
    re-sorted -- because a bucket's own bootstrap CI below is order-sensitive for a fixed seed
    exactly the way ``cluster_bootstrap_mean_diff`` and ``run_experiment.bootstrap_ci_mean_diff``
    both are.
    """
    counts: dict[str | None, int] = {}
    keyed: list[tuple[str, str | None]] = []
    for m in sorted(models):
        for item_id, conds in models[m].items():
            attr = item_attribute(conds)
            key = f"{m}::{item_id}"
            keyed.append((key, attr))
            counts[attr] = counts.get(attr, 0) + 1
    buckets: dict[str, list[str]] = {}
    for key, attr in keyed:
        bucket = attr if attr is not None and counts.get(attr, 0) >= min_rows else "other"
        buckets.setdefault(bucket, []).append(key)
    return buckets


def stratified_contrast(
    models: dict, a: str, b: str, min_rows: int = MIN_ROWS_PER_RELATION
) -> dict[str, dict]:
    """The discrete and continuous ``a-b`` contrast computed separately within each relation
    bucket from ``relation_buckets`` -- same primitives as the pooled contrast, just run on a
    row-order-preserving subset instead of the whole population."""
    all_items = {f"{m}::{i}": conds for m in sorted(models) for i, conds in models[m].items()}
    buckets = relation_buckets(models, min_rows)
    out: dict[str, dict] = {}
    for bucket_name, keys in buckets.items():
        sub = {k: all_items[k] for k in keys}
        disc = discrete_outcomes(sub)
        cont = {
            k: {c: {"delta_p_edited": row.get("delta_p_edited")} for c, row in conds.items()}
            for k, conds in sub.items()
        }
        out[bucket_name] = {
            "n_rows": len(keys),
            "discrete": discrete_contrast(disc, a, b),
            "continuous": paired_mean(cont, a, b, "delta_p_edited", seed=SEED),
        }
    return out


# =============================================================== 5. control-arm provenance


# How many records the offline dataset load asks for. The corpus this project draws from has
# under 12,600 rows total, so any scan comfortably above that pulls the whole thing regardless
# of the number requested -- the stream simply ends early. Kept as one constant rather than
# reproducing each part's own original --scan value because build_items truncates to n_items
# deterministically: the first N items of a bigger scan are byte-identical to the first N items
# of a scan sized exactly N, so over-scanning changes nothing about which items 1..N are.
PROVENANCE_SCAN = 50000

# The corpus each part's own repair search actually drew R2/R4 candidates from. This is NOT
# always the same size as the item set stage 2 processed: Part A replayed run 2's stage-1
# responses, in three separate per-model invocations of run_experiment.py, and every one of
# those three invocations still built its own top-level, 400-item corpus index before replaying
# that model's stage-1 rows onto a separately-rebuilt, larger item set -- an artefact of how run
# 3 replayed run 2's elicitations, discovered by checking rather than assumed: every built item
# in every part reproduces its own stored r1_total/r2_total exactly when the corpus is built at
# these sizes, and does not at other sizes tried (400/1200/1800/2000 were all tried against
# Part A; only 400 reproduces every stored pool size). Parts B and C generated fresh in one pass
# each, so their corpus and their item set are the same size.
CORPUS_N_ITEMS = {"A": 400, "B": 1200, "C": 1800}

# The option count each part's items were built with -- distractor selection (and therefore an
# item's own content beyond its id) depends on this, so it must match the run exactly.
PART_N_OPTIONS = {"A": 4, "B": 4, "C": 6}

# How many items to reconstruct for *looking up an item's own content* (its options, titles,
# profiles), as opposed to CORPUS_N_ITEMS above, which is how many items the R2/R4 candidate
# search itself drew from. This must exceed the largest attempted pool across all three parts --
# Part C alone attempted 1,293 items at stage two, by the same (model, item_id) join rule
# `analyze_run3.gate_eight_skew` uses to count an "attempted" row -- or a subset of built items
# that happen to sort earliest in the corpus would resolve while the rest silently could not be
# found at all, which would still show "100% resolved" on its own reduced, biased population.
# 2,000 clears the largest attempted pool with headroom.
ITEM_LOOKUP_N_ITEMS = 2000


def build_part5_items(part: str, scan: int = PROVENANCE_SCAN) -> tuple[dict, dict]:
    """``(items_by_id, corpus_index)`` for one part's control-provenance resolution: a lookup
    large enough to hold any built item's own content (``ITEM_LOOKUP_N_ITEMS``), and the corpus
    index the real run's own R1/R2 search drew from (``CORPUS_N_ITEMS[part]``). The corpus is a
    byte-identical prefix of the lookup set -- ``data.build_item`` never depends on how many
    items are requested, only on the record and the option count -- so one dataset load and one
    ``build_items`` call serves both."""
    records = load_records(C.DATASET_ID, C.DATASET_SPLIT, scan)
    n_options = PART_N_OPTIONS[part]
    items = build_items(records, n_items=ITEM_LOOKUP_N_ITEMS, n_options=n_options, seed=C.SEED)
    items_by_id = {it.item_id: it for it in items}
    corpus_index = repair.build_corpus_index(items[: CORPUS_N_ITEMS[part]])
    return items_by_id, corpus_index


def resolve_control_provenance(results: Path, part: str, *, items_by_id=None,
                                corpus_index=None) -> dict:
    """For every built item in ``part``, recover the accepted R1 source and R2/R4 control
    source by rebuilding the project's own ordered candidate lists
    (``repair._r1_candidates_ordered``, ``repair._r2_candidate_pool``,
    ``repair._r2_candidates_ordered``) and indexing them with the stage-2 records' own 1-based
    ``r1_examined``/``r2_examined``. No search is replayed -- only the winning position in an
    already-deterministic ordering is looked up.

    Refuses to compute a rate (``cross_item_rate``, ``relation_rates``) if any row is
    unresolved: a partial resolution is not a random sample of the built population (see
    ``ITEM_LOOKUP_N_ITEMS``), so a percentage over it would silently describe whichever items
    happened to be reachable rather than the built population the paper reports on.
    """
    if items_by_id is None or corpus_index is None:
        items_by_id, corpus_index = build_part5_items(part)
    directory, _ = PARTS[part]

    resolved = unresolved = same_item = cross_item = 0
    unresolved_reasons: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    same_relation_as_named = 0
    r1_pool_sizes: list[int] = []
    r2_pool_sizes: list[int] = []

    def bump(reason: str) -> None:
        nonlocal unresolved
        unresolved += 1
        unresolved_reasons[reason] = unresolved_reasons.get(reason, 0) + 1

    for path in sorted((results / directory).glob("stage2_*.jsonl")):
        for row in _rows(path):
            if not row.get("built"):
                continue
            item = items_by_id.get(row["item_id"])
            if item is None:
                bump("item_id_not_in_lookup")
                continue
            rival_title, attribute = row["rival_title"], row["attribute"]
            rejection = Rejection(sentence="", letter="", title=rival_title,
                                   matched_by="title", attribute=attribute)

            r1_ordered = repair._r1_candidates_ordered(item, rejection, corpus_index)
            r1_idx = row["r1_examined"] - 1
            if not (0 <= r1_idx < len(r1_ordered)):
                bump("r1_examined_out_of_range")
                continue
            source_entity, source_sentence, _stratum = r1_ordered[r1_idx]
            r1_sentence = repair.retarget(source_sentence, source_entity.title, rival_title)
            if r1_sentence is None:
                bump("r1_sentence_not_retargetable")
                continue

            target_len = repair._token_len(r1_sentence)
            r2_pool = repair._r2_candidate_pool(item, rival_title, attribute, corpus_index)
            r2_ordered = repair._r2_candidates_ordered(r2_pool, target_len)
            r2_idx = row["r2_examined"] - 1
            if not (0 <= r2_idx < len(r2_ordered)):
                bump("r2_examined_out_of_range")
                continue
            r2_attr, r2_title, _r2_sentence = r2_ordered[r2_idx]

            r1_pool_sizes.append(row["r1_total"])
            r2_pool_sizes.append(row["r2_total"])
            item_titles = {repair._norm(o.title) for o in item.options}
            if repair._norm(r2_title) in item_titles:
                same_item += 1
            else:
                cross_item += 1
            relation_counts[r2_attr] = relation_counts.get(r2_attr, 0) + 1
            if r2_attr == attribute:
                same_relation_as_named += 1
            resolved += 1

    fully_resolved = unresolved == 0 and resolved > 0
    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "unresolved_reasons": unresolved_reasons,
        "same_item": same_item,
        "cross_item": cross_item,
        "cross_item_rate": (cross_item / resolved) if fully_resolved else None,
        "relation_counts": relation_counts,
        "relation_rates": ({k: v / resolved for k, v in relation_counts.items()}
                            if fully_resolved else None),
        "same_relation_as_named": same_relation_as_named,
        "r1_pool_median": statistics.median(r1_pool_sizes) if r1_pool_sizes else None,
        "r1_pool_max": max(r1_pool_sizes) if r1_pool_sizes else None,
        "r2_pool_median": statistics.median(r2_pool_sizes) if r2_pool_sizes else None,
        "r2_pool_max": max(r2_pool_sizes) if r2_pool_sizes else None,
    }


# ========================================================== 6. fluency conditioning (E1)


# ``pilot/surprisal.py`` writes one file per model, named from the model path with every
# path separator turned into an underscore -- never one file per part, so a single glob picks
# up every model's rows across every part it was run on.
SURPRISAL_GLOB = "surprisal___*.jsonl"

# The parts this project's committed ``code/results/surprisal___*.jsonl`` actually cover --
# verified by reading the files, not assumed: every row's own ``part`` field is either "A" or
# "C" (Part B, the second 4-option roster, was never run through E1). A part requested on the
# CLI that is not in this tuple contributes nothing to this section and is reported as skipped
# rather than silently producing an empty, unexplained table.
FLUENCY_PARTS = ("A", "C")

# The two content contrasts round 5's own fluency question is about -- R3-R4 first, matching
# the committed fluency-confound analysis, which calls it "the paper's spine".
# R2-R4 (the location/content interaction) has no fluency question of its own: R2 and R4 are
# already both length-matched control insertions, so there is no relevant/irrelevant fluency
# gap to condition on -- excluded on purpose, not by oversight.
FLUENCY_CONTRASTS = (("R3", "R4"), ("R1", "R2"))

# Population size below which a leave-one-model-out drop is not reported: a tercile already
# holds under 300 pairs, and a further drop can leave fewer discordant pairs than a McNemar
# test can say anything useful about. No such floor is applied elsewhere in this module --
# it exists here only because this section's own leave-one-out is nested two levels deep
# (contrast x tercile x drop) and an empty or near-empty cell would otherwise print a
# meaningless odds ratio next to the real ones.
FLUENCY_LOO_MIN_DISCORDANT = 1


def _dedup_surprisal_rows(rows: Iterable[dict]) -> tuple[dict[tuple[str, str, str, str], dict], int]:
    """Last-write-wins de-duplication on ``(model, part, item_id, condition)``.

    ``pilot/surprisal.py`` appends to its per-model file every time it runs (``open(path, "a")``
    in ``run_surprisal``); a model re-run after a partial or aborted pass therefore leaves two
    writes for the same key in the same file, not one overwritten write. Verified directly
    against this project's own committed files rather than assumed (rule 5): as of this section
    being written, the three committed ``surprisal___*.jsonl`` carry zero such duplicates, but
    the loader still de-duplicates unconditionally and reports the count, since a future re-run
    appending more rows would otherwise silently double-count an item the next time this module
    runs. Keeps the *last* row for a repeated key, on the theory a later write reflects a more
    complete or more recent read of that (item, condition) than an earlier one.
    """
    out: dict[tuple[str, str, str, str], dict] = {}
    n_dropped = 0
    for row in rows:
        key = (row["model"], row["part"], row["item_id"], row["condition"])
        if key in out:
            n_dropped += 1
        out[key] = row
    return out, n_dropped


def load_surprisal(results: Path) -> tuple[dict[tuple[str, str, str, str], dict], int]:
    """``({(model, part, item_id, condition): row}, n_duplicates_dropped)`` over every
    ``surprisal___*.jsonl`` under ``results``, in sorted file order (so re-runs across files,
    not just within one, still resolve last-write-wins deterministically)."""

    def _iter_rows():
        for path in sorted(results.glob(SURPRISAL_GLOB)):
            yield from _rows(path)

    return _dedup_surprisal_rows(_iter_rows())


def load_fluency_choices(
    results: Path, parts: Iterable[str]
) -> dict[tuple[str, str, str, str], bool | None]:
    """``chosen_is_edited`` keyed exactly like ``load_surprisal``'s rows, read from each part's
    own committed ``stage3_*.jsonl`` -- stage 2 never carries the discrete choice outcome, only
    whether repair built cleanly."""
    out: dict[tuple[str, str, str, str], bool | None] = {}
    for part in parts:
        directory, _ = PARTS[part]
        for path in sorted((results / directory).glob("stage3_*.jsonl")):
            for row in _rows(path):
                key = (row["model"], part, row["item_id"], row["condition"])
                out[key] = row.get("chosen_is_edited")
    return out


def join_fluency_contrast(
    surp: dict[tuple[str, str, str, str], dict],
    choices: dict[tuple[str, str, str, str], bool | None],
    a: str, b: str,
) -> list[dict]:
    """One row per ``(model, part, item_id)`` with a usable pair on contrast ``a-b``: both
    conditions' surprisal read must be ``complete`` (a non-empty inserted span with every token
    scored) *and* both conditions' ``chosen_is_edited`` must be non-null -- the join rule this
    section's own docstring specifies, applied uniformly to every statistic below (condition
    means, the paired contrast, and the terciles) rather than a broader population for one and a
    narrower one for another.

    Returned in ``sorted(by_item.items())`` order (i.e. sorted by the ``(model, part, item_id)``
    key) so a caller that resorts by ``delta_nll`` for the terciles gets a fully deterministic
    tie-break -- ties in ``delta_nll`` are not vanishingly rare with only three decimal places of
    real signal, and a resort that is not itself stable-input would silently change which item
    lands in which tercile from one run to the next.
    """
    by_item: dict[tuple[str, str, str], dict[str, dict]] = {}
    for (model, part, item_id, cond), row in surp.items():
        if cond not in (a, b):
            continue
        by_item.setdefault((model, part, item_id), {})[cond] = row

    joined: list[dict] = []
    for key, conds in sorted(by_item.items()):
        ra, rb = conds.get(a), conds.get(b)
        if ra is None or rb is None or not ra.get("complete") or not rb.get("complete"):
            continue
        model, part, item_id = key
        ca = choices.get((model, part, item_id, a))
        cb = choices.get((model, part, item_id, b))
        if ca is None or cb is None:
            continue
        joined.append({
            "key": key, "model": model, "part": part, "item_id": item_id,
            "mean_nll_a": ra["mean_nll"], "mean_nll_b": rb["mean_nll"],
            "delta_nll": ra["mean_nll"] - rb["mean_nll"],
            "chosen_a": bool(ca), "chosen_b": bool(cb),
        })
    return joined


def fluency_condition_means(joined: list[dict]) -> dict:
    """Per-model and pooled mean NLL for each of the contrast's two conditions, over exactly the
    joined (paired-and-choice-complete) population the rest of this section uses -- not the
    broader, unpaired population ``pilot/surprisal.py``'s own ``condition_summary`` reports, so
    every number in this section's table is read off one consistent population."""
    by_model: dict[str, dict[str, list[float]]] = {}
    for row in joined:
        m = by_model.setdefault(row["model"], {"a": [], "b": []})
        m["a"].append(row["mean_nll_a"])
        m["b"].append(row["mean_nll_b"])

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    per_model = {
        short_model(m): {"n": len(v["a"]), "mean_a": _mean(v["a"]), "mean_b": _mean(v["b"])}
        for m, v in sorted(by_model.items())
    }
    pooled_a = [row["mean_nll_a"] for row in joined]
    pooled_b = [row["mean_nll_b"] for row in joined]
    pooled = {"n": len(joined), "mean_a": _mean(pooled_a), "mean_b": _mean(pooled_b)}
    return {"per_model": per_model, "pooled": pooled}


def fluency_pooled_contrast(joined: list[dict], *, seed: int = SEED) -> dict:
    """Pooled paired mean of ``delta_nll`` with a bootstrap CI, plus the same per model --
    ``run_experiment.bootstrap_ci_mean_diff``, the same one every other paired mean in this
    project uses."""
    pooled = bootstrap_ci_mean_diff(
        [row["delta_nll"] for row in joined], seed=seed, n_resamples=RESAMPLES
    )
    by_model: dict[str, list[float]] = {}
    for row in joined:
        by_model.setdefault(row["model"], []).append(row["delta_nll"])
    per_model = {
        short_model(m): bootstrap_ci_mean_diff(diffs, seed=seed, n_resamples=RESAMPLES)
        for m, diffs in sorted(by_model.items())
    }
    return {"pooled": pooled, "per_model": per_model}


def fluency_tercile_bounds(n: int) -> list[int]:
    """Boundary indices ``[0, floor(n/3), floor(2n/3), n]`` for cutting a list of length ``n``
    sorted ascending into three terciles.

    Not the more obvious "``n // 3`` per tercile, remainder to the first groups" split: that
    puts a size-``n%3`` surplus on the *widest-gap* terciles, which for this project's own
    881-row R3-R4 join changes T1 from 293 to 294 items and silently shifts which items fall in
    T1 vs T3 relative to the released analysis. ``floor(i*n/3)`` boundaries instead put the
    surplus on the later terciles and were checked to reproduce the released T1/T2/T3 sizes
    (293/294/294) and every discordant pair count in them exactly.
    """
    return [i * n // 3 for i in range(4)]


def fluency_terciles(joined: list[dict]) -> list[list[dict]]:
    """``joined`` split into three terciles of ascending ``delta_nll`` (T1 = widest content-arm
    advantage, i.e. most negative; T3 = arms closest in surprisal), using
    ``fluency_tercile_bounds``. Sorted by ``(delta_nll, key)`` -- the same deterministic tie-break
    ``join_fluency_contrast`` already orders its own output by, made explicit again here since a
    tercile split is exactly the place a silent tie-break inconsistency would change results."""
    ordered = sorted(joined, key=lambda row: (row["delta_nll"], row["key"]))
    bounds = fluency_tercile_bounds(len(ordered))
    return [ordered[bounds[i]:bounds[i + 1]] for i in range(3)]


def _fluency_outcomes(rows: list[dict], a: str, b: str) -> dict[str, dict[str, bool]]:
    """``rows`` (this section's own joined-row shape) reshaped into the
    ``{key: {condition: chosen_is_edited}}`` form ``analyze_run3.discrete_contrast`` (and
    therefore the project's own ``mcnemar``/``odds_ratio``) already consumes."""
    return {
        f"{row['model']}::{row['part']}::{row['item_id']}": {a: row["chosen_a"], b: row["chosen_b"]}
        for row in rows
    }


def fluency_tercile_discrete(rows: list[dict], a: str, b: str) -> dict:
    """The discrete ``a-b`` contrast (McNemar b/c, exact p, odds ratio) within one tercile,
    via ``analyze_run3.discrete_contrast`` -- the project's own ``mcnemar`` and ``odds_ratio``,
    not a re-derivation."""
    return discrete_contrast(_fluency_outcomes(rows, a, b), a, b)


def fluency_loo_within_tercile(rows: list[dict], a: str, b: str) -> list[dict]:
    """The tercile's own discrete contrast, then again with each model that appears in it
    dropped in turn -- same "dropped: None first, then each model" shape
    ``leave_one_model_out`` (section 1) already reports, so a reader who has seen that table
    reads this one for free."""
    models = sorted({row["model"] for row in rows})
    out = [{"dropped": None, "n_models": len(models),
            "discrete": fluency_tercile_discrete(rows, a, b)}]
    for model in models:
        remaining = [row for row in rows if row["model"] != model]
        out.append({
            "dropped": short_model(model),
            "n_models": len(models) - 1,
            "discrete": fluency_tercile_discrete(remaining, a, b),
        })
    return out


def fluency_contrast_report(surp: dict, choices: dict, a: str, b: str) -> dict:
    """Everything this section reports for one contrast: condition means, the pooled paired
    contrast, and the tercile split with its own discrete contrast and leave-one-out."""
    joined = join_fluency_contrast(surp, choices, a, b)
    terciles = fluency_terciles(joined)
    return {
        "contrast": f"{a}-{b}",
        "n_joined": len(joined),
        "condition_means": fluency_condition_means(joined),
        "pooled_contrast": fluency_pooled_contrast(joined),
        "terciles": [
            {
                "tercile": i + 1,
                "n": len(t),
                "delta_nll_range": (
                    [t[0]["delta_nll"], t[-1]["delta_nll"]] if t else [None, None]
                ),
                "discrete": fluency_tercile_discrete(t, a, b),
                "leave_one_out": fluency_loo_within_tercile(t, a, b),
            }
            for i, t in enumerate(terciles)
        ],
    }


# ==================================================================================== reporting


def _fmt_ci(d: dict, places: int = 4) -> str:
    if d.get("ci_low") is None:
        return "n/a"
    return f"{d['mean']:+.{places}f} [{d['ci_low']:+.{places}f}, {d['ci_high']:+.{places}f}]"


def _fmt_p(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.4f}"


def report_loo(results: Path, parts: list[str]) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 1. Leave-one-model-out")
    w("")
    w("Both measures, pooled over all models and then again with each model dropped, for the "
      "three contrasts round 5 asks about, per part.")
    w("")
    for part in parts:
        models = load_part(results, part)
        payload[part] = {}
        w(f"### Part {part}")
        w("")
        w("| contrast | dropped | disc b | disc c | OR | exact p | cont n | delta_p mean [95% CI] |")
        w("|---|---|---:|---:|---:|---:|---:|---|")
        for a, b in ROUND5_CONTRASTS:
            rows = leave_one_model_out(models, a, b)
            payload[part][f"{a}-{b}"] = rows
            for row in rows:
                dropped = row["dropped"] or "-- (all models)"
                d, cm = row["discrete"], row["continuous"]
                w(f"| {a}-{b} | {dropped} | {d['b_a_only']} | {d['c_b_only']} "
                  f"| {d['or']['or']:.2f} | {_fmt_p(d['p_exact_two_sided'])} | {cm['n']} "
                  f"| {_fmt_ci(cm)} |")
        w("")
    return "\n".join(lines) + "\n", payload


def report_heterogeneity(results: Path, parts: list[str]) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 2. Between-model heterogeneity")
    w("")
    w("Cochran's Q over the three per-model log odds ratios, per contrast per part. tau^2 is "
      "not reported -- see `heterogeneity`'s docstring.")
    w("")
    w("| part | contrast | Q | df | p | I^2 |")
    w("|---|---|---:|---:|---:|---:|")
    for part in parts:
        models = load_part(results, part)
        payload[part] = {}
        for a, b in ROUND5_CONTRASTS:
            h = heterogeneity(models, a, b)
            payload[part][f"{a}-{b}"] = h
            q = "n/a" if h["q"] is None else f"{h['q']:.2f}"
            i2 = "n/a" if h["i2"] is None else f"{h['i2']:.1f}%"
            w(f"| {part} | {a}-{b} | {q} | {h['df']} | {_fmt_p(h['p'])} | {i2} |")
    w("")
    return "\n".join(lines) + "\n", payload


def report_cluster(results: Path, parts: list[str]) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 3. Item-clustered bootstrap")
    w("")
    w("The same paired contrasts as section 1's pooled row, resampling distinct item ids as "
      "clusters instead of rows. A row-clustered and an item-clustered interval agreeing is a "
      "check on the pooled-row bootstrap the rest of this project's reports use by default.")
    w("")
    for part in parts:
        models = load_part(results, part)
        n_built_ids = distinct_built_item_ids(results, part)
        payload[part] = {"distinct_built_item_ids": n_built_ids}
        w(f"### Part {part} ({n_built_ids} distinct built item ids)")
        w("")
        w("| contrast | measure | n rows | n clusters | mean [95% CI] |")
        w("|---|---|---:|---:|---|")
        for a, b in ROUND5_CONTRASTS:
            cont_clusters = _cluster_diffs_continuous(models, a, b)
            disc_clusters = _cluster_diffs_discrete(models, a, b)
            cont_ci = cluster_bootstrap_mean_diff(cont_clusters, seed=SEED, n_resamples=RESAMPLES)
            disc_ci = cluster_bootstrap_mean_diff(disc_clusters, seed=SEED, n_resamples=RESAMPLES)
            payload[part][f"{a}-{b}"] = {"delta_p_edited": cont_ci, "chosen_is_edited": disc_ci}
            w(f"| {a}-{b} | delta_p_edited | {cont_ci['n']} | {cont_ci['n_clusters']} "
              f"| {_fmt_ci(cont_ci)} |")
            w(f"| {a}-{b} | chosen_is_edited (risk diff.) | {disc_ci['n']} "
              f"| {disc_ci['n_clusters']} | {_fmt_ci(disc_ci)} |")
        w("")
    return "\n".join(lines) + "\n", payload


def report_stratify(results: Path, parts: list[str]) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 4. Per-relation stratification")
    w("")
    w(f"The same three contrasts, computed separately within each named-attribute relation with "
      f"at least {MIN_ROWS_PER_RELATION} (model, item) pairs; every smaller relation is pooled "
      f"as `other`.")
    w("")
    for part in parts:
        models = load_part(results, part)
        buckets = relation_buckets(models)
        payload[part] = {"bucket_sizes": {k: len(v) for k, v in buckets.items()}}
        w(f"### Part {part} (buckets: "
          f"{', '.join(f'{k}={len(v)}' for k, v in sorted(buckets.items()))})")
        w("")
        w("| contrast | relation | disc b | disc c | OR | exact p | cont n | delta_p mean [95% CI] |")
        w("|---|---|---:|---:|---:|---:|---:|---|")
        for a, b in ROUND5_CONTRASTS:
            strat = stratified_contrast(models, a, b)
            payload[part][f"{a}-{b}"] = strat
            for bucket_name in sorted(strat):
                s = strat[bucket_name]
                d, cm = s["discrete"], s["continuous"]
                w(f"| {a}-{b} | {bucket_name} | {d['b_a_only']} | {d['c_b_only']} "
                  f"| {d['or']['or']:.2f} | {_fmt_p(d['p_exact_two_sided'])} | {cm['n']} "
                  f"| {_fmt_ci(cm)} |")
        w("")
    return "\n".join(lines) + "\n", payload


def report_provenance(results: Path, parts: list[str]) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 5. Control-arm provenance")
    w("")
    w("For every built item, whether the accepted R2/R4 control sentence names an entity "
      "already inside that item or one from elsewhere in the corpus, the R1 and R2 candidate "
      "pool sizes, and the distribution of the control sentence's own relation.")
    w("")
    w("| part | built | resolved | unresolved | R1 pool (median, max) | R2 pool (median, max) |")
    w("|---|---:|---:|---:|---|---|")
    provenance: dict[str, dict] = {}
    for part in parts:
        prov = resolve_control_provenance(results, part)
        provenance[part] = prov
        built = prov["resolved"] + prov["unresolved"]
        w(f"| {part} | {built} | {prov['resolved']} | {prov['unresolved']} "
          f"| {prov['r1_pool_median']}, {prov['r1_pool_max']} "
          f"| {prov['r2_pool_median']}, {prov['r2_pool_max']} |")
    w("")
    w("| part | R2 source same-item | R2 source cross-item | same relation as named attribute |")
    w("|---|---|---|---|")
    for part in parts:
        prov = provenance[part]
        payload[part] = prov
        if prov["cross_item_rate"] is None:
            w(f"| {part} | {prov['same_item']} | {prov['cross_item']} "
              f"| {prov['same_relation_as_named']} -- **percentage withheld: "
              f"{prov['unresolved']} row(s) unresolved** |")
            continue
        w(f"| {part} | {prov['same_item']} ({1 - prov['cross_item_rate']:.1%}) "
          f"| {prov['cross_item']} ({prov['cross_item_rate']:.1%}) "
          f"| {prov['same_relation_as_named']} of {prov['resolved']} |")
    w("")
    w("Control sentence's own relation, full distribution (not a top-k):")
    w("")
    for part in parts:
        prov = provenance[part]
        w(f"**Part {part}**")
        if prov["relation_rates"] is None:
            w(f"- percentage withheld: {prov['unresolved']} row(s) unresolved")
            w("")
            continue
        for rel, n in sorted(prov["relation_counts"].items(), key=lambda kv: -kv[1]):
            w(f"- `{rel}`: {n} ({prov['relation_rates'][rel]:.1%})")
        w(f"- distinct relations: {len(prov['relation_counts'])}")
        w("")
    return "\n".join(lines) + "\n", payload


def report_fluency(results: Path, parts: list[str]) -> tuple[str, dict]:
    """Whether the R3-R4 and R1-R2 content contrasts survive conditioning on the fluency gap
    ``pilot/surprisal.py`` measures: per-model/pooled mean NLL by condition, the paired contrast,
    and -- within terciles of paired Delta-surprisal -- the discrete contrast and its
    leave-one-model-out.

    Pooled across every part in ``parts`` that ``FLUENCY_PARTS`` actually covers (the section is
    not run per part the way sections 1-4 are: E1 was run once per model against the union of
    Part A and Part C's committed records, not once per part, and splitting it back apart would
    shrink every tercile below where a McNemar test says anything). A requested part outside
    ``FLUENCY_PARTS`` is named and skipped rather than silently ignored.
    """
    lines: list[str] = []
    w = lines.append
    payload: dict = {}
    w("## 6. Fluency conditioning (E1)")
    w("")
    w("Per-item surprisal from `pilot/surprisal.py`'s own `surprisal___*.jsonl`, joined to the "
      "committed stage-3 `chosen_is_edited`. Kept only where both conditions of a contrast have "
      "a non-null surprisal read and a non-null choice.")
    w("")

    used_parts = [p for p in parts if p in FLUENCY_PARTS]
    skipped_parts = [p for p in parts if p not in FLUENCY_PARTS]
    payload["parts_used"] = used_parts
    payload["parts_skipped"] = skipped_parts
    if skipped_parts:
        w(f"Part(s) {', '.join(skipped_parts)} requested but not covered by any committed "
          f"`surprisal___*.jsonl` (only {'/'.join(FLUENCY_PARTS)} are) -- skipped, not silently "
          "pooled in as zero rows.")
        w("")
    if not used_parts:
        w("No requested part has surprisal coverage; nothing to report.")
        w("")
        payload["n_duplicates_dropped"] = 0
        payload["contrasts"] = {}
        return "\n".join(lines) + "\n", payload

    surp, n_dup = load_surprisal(results)
    surp = {k: v for k, v in surp.items() if k[1] in used_parts}
    choices = load_fluency_choices(results, used_parts)
    payload["n_duplicates_dropped"] = n_dup
    w(f"{n_dup} duplicate `(model, part, item_id, condition)` write(s) dropped "
      f"(last write kept) before anything below.")
    w("")

    payload["contrasts"] = {}
    for a, b in FLUENCY_CONTRASTS:
        rep = fluency_contrast_report(surp, choices, a, b)
        payload["contrasts"][f"{a}-{b}"] = rep

        w(f"### {a}-{b}")
        w("")
        w(f"n joined = {rep['n_joined']}")
        w("")
        w("**Mean NLL by condition**")
        w("")
        w(f"| model | n | mean {a} | mean {b} |")
        w("|---|---:|---:|---:|")
        cm = rep["condition_means"]
        for model, d in cm["per_model"].items():
            w(f"| {model} | {d['n']} | {d['mean_a']:.3f} | {d['mean_b']:.3f} |")
        pooled_cm = cm["pooled"]
        w(f"| **pooled** | {pooled_cm['n']} | {pooled_cm['mean_a']:.3f} "
          f"| {pooled_cm['mean_b']:.3f} |")
        w("")
        w(f"**Paired contrast ({a}-{b})**")
        w("")
        w("| model | n | mean diff [95% CI] |")
        w("|---|---:|---|")
        pc = rep["pooled_contrast"]
        for model, d in pc["per_model"].items():
            w(f"| {model} | {d['n']} | {_fmt_ci(d, places=3)} |")
        w(f"| **pooled** | {pc['pooled']['n']} | {_fmt_ci(pc['pooled'], places=3)} |")
        w("")

        w(f"**Terciles of paired Delta-surprisal ({a}-{b})**")
        w("")
        w("| tercile | range | n | disc b | disc c | OR | exact p |")
        w("|---|---|---:|---:|---:|---:|---:|")
        for t in rep["terciles"]:
            lo, hi = t["delta_nll_range"]
            rng = "n/a" if lo is None else f"[{lo:+.2f}, {hi:+.2f}]"
            d = t["discrete"]
            w(f"| T{t['tercile']} | {rng} | {t['n']} | {d['b_a_only']} | {d['c_b_only']} "
              f"| {d['or']['or']:.2f} | {_fmt_p(d['p_exact_two_sided'])} |")
        w("")

        w(f"**Leave-one-model-out within each tercile ({a}-{b})**")
        w("")
        w("| tercile | dropped | n models | disc b | disc c | OR | exact p |")
        w("|---|---|---:|---:|---:|---:|---:|")
        for t in rep["terciles"]:
            for row in t["leave_one_out"]:
                dropped = row["dropped"] or "-- (all models)"
                d = row["discrete"]
                w(f"| T{t['tercile']} | {dropped} | {row['n_models']} "
                  f"| {d['b_a_only']} | {d['c_b_only']} | {d['or']['or']:.2f} "
                  f"| {_fmt_p(d['p_exact_two_sided'])} |")
        w("")

    return "\n".join(lines) + "\n", payload


SECTIONS = {
    "loo": report_loo,
    "heterogeneity": report_heterogeneity,
    "cluster": report_cluster,
    "stratify": report_stratify,
    "provenance": report_provenance,
    "fluency": report_fluency,
}


def build_report(results: Path, parts: list[str], sections: list[str]) -> tuple[str, dict]:
    lines: list[str] = []
    w = lines.append
    w("# Round-5 offline analyses")
    w("")
    w("Generated by `python -m pilot.analyze_round5` from the committed stage-2/stage-3 JSONL "
      f"in `code/results/exp3a`, `exp3b`, `exp3c`. Every bootstrap here uses seed `{SEED}` with "
      f"`{RESAMPLES}` resamples, matching `pilot/analyze_run3.py`; every exact test is the same "
      "`math.comb`-based binomial `run_experiment.mcnemar` uses.")
    w("")
    payload: dict = {"seed": SEED, "n_resamples": RESAMPLES, "parts": parts,
                      "sections": sections, "results": {}}
    for section in sections:
        text, section_payload = SECTIONS[section](results, parts)
        lines.append(text)
        payload["results"][section] = section_payload
    return "\n".join(lines), payload


# ==================================================================================== CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=C.RESULTS_DIR, type=Path,
                     help="the results tree holding exp3a, exp3b, exp3c "
                          f"(default: {C.RESULTS_DIR})")
    ap.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--section", choices=[*SECTIONS, "all"], default="all")
    ap.add_argument("--out", default=None, type=Path,
                     help="where to write the summary JSON (default: a fixed path under "
                          "config.RESULTS_DIR; refuses to overwrite an existing file)")
    args = ap.parse_args(argv)

    parts = ["A", "B", "C"] if args.part == "all" else [args.part]
    sections = list(SECTIONS) if args.section == "all" else [args.section]

    report, payload = build_report(args.results, parts, sections)
    print(report)

    models_by_part = {
        part: sorted(short_model(m) for m in load_part(args.results, part))
        for part in parts
    }
    summary = {
        "generated_utc": C.stamp_utc(),
        "seed": SEED,
        "n_resamples": RESAMPLES,
        "parts": parts,
        "sections": sections,
        "models": models_by_part,
    }

    out = args.out or (C.RESULTS_DIR / "analyze_round5_summary.json")
    if out.exists():
        print(f"refusing to overwrite existing results file: {out}")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
