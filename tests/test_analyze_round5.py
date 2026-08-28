"""Tests for pilot.analyze_round5: recomputes round-5 statistics from hand-written fixtures.

No GPU, no network; the provenance-resolver tests pass hand-built Entity/Item objects and a
hand-built corpus index in directly (via resolve_control_provenance's items_by_id/corpus_index
kwargs) rather than loading the real dataset, so they run in seconds like every other test here.
"""

from __future__ import annotations

import json
import math

from pilot.data import Entity, Item
from pilot.repair import build_corpus_index

from pilot.analyze_round5 import (
    MIN_ROWS_PER_RELATION,
    _chi2_sf_even_df,
    _dedup_surprisal_rows,
    _log_or,
    cluster_bootstrap_mean_diff,
    fluency_loo_within_tercile,
    fluency_tercile_bounds,
    fluency_terciles,
    fluency_tercile_discrete,
    heterogeneity,
    item_attribute,
    join_fluency_contrast,
    load_fluency_choices,
    relation_buckets,
    resolve_control_provenance,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- Cochran's Q and I^2 (2. heterogeneity)


def test_chi2_sf_even_df_at_zero_is_one():
    assert _chi2_sf_even_df(0.0, 2) == 1.0


def test_chi2_sf_even_df_df2_matches_the_exponential_closed_form():
    """A chi-squared variable with 2 degrees of freedom is exactly Exponential(rate=1/2), so
    its survival function is exp(-x/2) -- the identity _chi2_sf_even_df's df=2 case reduces to,
    checked here independently of the general-df derivation."""
    for x in (0.5, 2.0, 5.0, 10.0):
        assert math.isclose(_chi2_sf_even_df(x, 2), math.exp(-x / 2.0), rel_tol=1e-12)


def test_chi2_sf_even_df_rejects_odd_df():
    import pytest

    with pytest.raises(ValueError):
        _chi2_sf_even_df(1.0, 3)


def test_log_or_uncorrected_matches_hand_computation():
    logor, var = _log_or(8, 2)
    assert math.isclose(logor, math.log(4.0), rel_tol=1e-12)
    assert math.isclose(var, 1.0 / 8 + 1.0 / 2, rel_tol=1e-12)


def test_log_or_applies_haldane_anscombe_only_on_a_zero_cell():
    logor, var = _log_or(0, 5)
    assert math.isclose(logor, math.log(0.5 / 5.5), rel_tol=1e-12)
    assert math.isclose(var, 1.0 / 0.5 + 1.0 / 5.5, rel_tol=1e-12)


def test_heterogeneity_on_a_hand_computable_toy_case():
    """Three models, discrete outcomes engineered so b/c are exactly (8,2), (8,2), (2,8) --
    worked by hand:

        model A: b=8, c=2 -> log OR = ln(4) = 1.386294, var = 1/8+1/2 = 0.625, weight = 1.6
        model B: b=8, c=2 -> identical to A
        model C: b=2, c=8 -> log OR = ln(0.25) = -1.386294, var = 0.625, weight = 1.6

    Weights are equal, so the pooled log OR is the plain mean of the three:
        pooled = (1.386294 + 1.386294 - 1.386294) / 3 = 0.462098

    Q = sum(w * (log_or - pooled)^2)
      = 1.6 * [(1.386294-0.462098)^2 * 2 + (-1.386294-0.462098)^2]
      ~= 8.199731 (full precision; the rounded 4-decimal inputs above give ~8.2007)

    df = 2 (three models). p = exp(-Q/2) ~= 0.016575.
    I^2 = (Q-df)/Q * 100 ~= 75.6%.

    Each model's outcomes below are built with exactly 10 items so b/c fall out of a plain
    2x2 discordant-pair count rather than being injected directly, exercising the same
    discrete_outcomes/discrete_contrast path the real report uses.
    """

    def make_outcomes(n_b: int, n_c: int, n_concordant_true: int, n_concordant_false: int):
        """n_b items where R3=True,R4=False (b); n_c where R3=False,R4=True (c); the rest
        concordant so they contribute nothing to b/c but keep item ids distinct."""
        items = {}
        i = 0
        for _ in range(n_b):
            items[f"i{i}"] = {"R3": {"R3": True}, "R4": {"R4": False}}
            i += 1
        for _ in range(n_c):
            items[f"i{i}"] = {"R3": {"R3": False}, "R4": {"R4": True}}
            i += 1
        for _ in range(n_concordant_true):
            items[f"i{i}"] = {"R3": {"R3": True}, "R4": {"R4": True}}
            i += 1
        for _ in range(n_concordant_false):
            items[f"i{i}"] = {"R3": {"R3": False}, "R4": {"R4": False}}
            i += 1
        return items

    # discrete_outcomes reads chosen_is_edited off each condition's own row; build rows with
    # exactly that shape via the same helper the real pipeline would (condition -> row dict).
    def rows_from(n_b, n_c):
        items = {}
        i = 0
        for _ in range(n_b):
            items[f"i{i}"] = {"R3": {"chosen_is_edited": True}, "R4": {"chosen_is_edited": False}}
            i += 1
        for _ in range(n_c):
            items[f"i{i}"] = {"R3": {"chosen_is_edited": False}, "R4": {"chosen_is_edited": True}}
            i += 1
        return items

    models = {
        "model-a": rows_from(8, 2),
        "model-b": rows_from(8, 2),
        "model-c": rows_from(2, 8),
    }
    h = heterogeneity(models, "R3", "R4")
    assert h["df"] == 2
    assert math.isclose(h["q"], 8.199731, rel_tol=1e-5)
    assert math.isclose(h["p"], 0.016575, rel_tol=1e-3)
    assert math.isclose(h["i2"], 75.609, abs_tol=0.01)
    assert len(h["per_model"]) == 3


def test_heterogeneity_is_zero_when_every_model_has_the_same_odds_ratio():
    def rows_from(n_b, n_c):
        items = {}
        i = 0
        for _ in range(n_b):
            items[f"i{i}"] = {"R3": {"chosen_is_edited": True}, "R4": {"chosen_is_edited": False}}
            i += 1
        for _ in range(n_c):
            items[f"i{i}"] = {"R3": {"chosen_is_edited": False}, "R4": {"chosen_is_edited": True}}
            i += 1
        return items

    models = {"a": rows_from(6, 3), "b": rows_from(6, 3), "c": rows_from(6, 3)}
    h = heterogeneity(models, "R3", "R4")
    assert math.isclose(h["q"], 0.0, abs_tol=1e-9)
    assert math.isclose(h["i2"], 0.0, abs_tol=1e-9)
    assert math.isclose(h["p"], 1.0, rel_tol=1e-9)


# --------------------------------------------------------- item-clustered bootstrap (3.)


def test_cluster_bootstrap_moves_a_multi_row_item_as_one_unit():
    """Two clusters: "many" contributes three identical rows (value 10.0 each, as if the same
    item had built and completed for three different models), "one" contributes a single row
    (value -10.0). Resampled means can therefore only take one of three values:

        both draws land on "many"  -> mean = 10.0
        one of each                -> mean = (3*10 - 10) / 4 = 5.0
        both draws land on "one"   -> mean = -10.0

    A *row*-level bootstrap over the pooled [10,10,10,-10] would also reach 0.0 and -5.0 (e.g.
    one row of -10 drawn among four total row-draws gives (3*10-10)/4=5 only if all three 10s
    are drawn together, but a plain row-level resample draws each of the four positions
    independently and can produce 2 negative draws out of 4 for a mean of 0.0, which no
    whole-cluster resample can ever produce here) -- so every resample mean landing in exactly
    {-10.0, 5.0, 10.0} and never at 0.0 or -5.0 is the fingerprint of cluster-level, not
    row-level, resampling.
    """
    clusters = {"many": [10.0, 10.0, 10.0], "one": [-10.0]}
    result = cluster_bootstrap_mean_diff(clusters, seed=1, n_resamples=500)

    assert result["n"] == 4
    assert result["n_clusters"] == 2
    assert math.isclose(result["mean"], (30.0 - 10.0) / 4.0, rel_tol=1e-12)
    possible = {-10.0, 5.0, 10.0}
    assert result["ci_low"] in possible
    assert result["ci_high"] in possible


def test_cluster_bootstrap_on_no_clusters_is_all_none():
    result = cluster_bootstrap_mean_diff({}, seed=1)
    assert result == {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_clusters": 0,
                       "seed": 1, "n_resamples": 10000}


def test_cluster_bootstrap_is_deterministic_under_a_fixed_seed():
    clusters = {"a": [0.1, 0.2], "b": [-0.3], "c": [0.05, 0.05, 0.05]}
    first = cluster_bootstrap_mean_diff(clusters, seed=7, n_resamples=300)
    second = cluster_bootstrap_mean_diff(clusters, seed=7, n_resamples=300)
    assert first == second


# ------------------------------------------------------ relation-bucketing threshold (4.)


def _cond_row(attribute):
    return {"R0": {"attribute": attribute}}


def test_item_attribute_reads_off_any_condition_row():
    conds = {"R0": {"attribute": None}, "R1": {"attribute": "director"}, "R2": {}}
    assert item_attribute(conds) == "director"


def test_item_attribute_is_none_when_no_condition_carries_one():
    conds = {"R0": {"attribute": None}, "R1": {}}
    assert item_attribute(conds) is None


def test_relation_buckets_keeps_relations_at_or_above_threshold_separate():
    """Three relations: "big" has 3 (model,item) pairs, "small" has 1, threshold is 2 -- so
    "big" must keep its own bucket and "small" must fold into "other"."""
    models = {
        "m1": {
            "i1": _cond_row("big"),
            "i2": _cond_row("big"),
            "i3": _cond_row("small"),
        },
        "m2": {
            "i1": _cond_row("big"),
        },
    }
    buckets = relation_buckets(models, min_rows=2)
    assert set(buckets) == {"big", "other"}
    assert len(buckets["big"]) == 3
    assert len(buckets["other"]) == 1
    assert "m1::i3" in buckets["other"]


def test_relation_buckets_threshold_is_inclusive_at_the_boundary():
    models = {"m1": {"i1": _cond_row("x"), "i2": _cond_row("x")}}
    exactly_at = relation_buckets(models, min_rows=2)
    just_above = relation_buckets(models, min_rows=3)
    assert set(exactly_at) == {"x"}
    assert set(just_above) == {"other"}


def test_relation_buckets_every_pooled_key_appears_exactly_once():
    models = {
        "m1": {"i1": _cond_row("a"), "i2": _cond_row("b")},
        "m2": {"i1": _cond_row("a"), "i3": _cond_row(None)},
    }
    buckets = relation_buckets(models, min_rows=1)
    all_keys = [k for keys in buckets.values() for k in keys]
    assert sorted(all_keys) == sorted(
        f"{m}::{i}" for m, its in models.items() for i in its
    )
    assert len(all_keys) == len(set(all_keys))


def test_min_rows_per_relation_default_is_a_positive_int():
    assert isinstance(MIN_ROWS_PER_RELATION, int)
    assert MIN_ROWS_PER_RELATION > 0


# ---------------------------------------------------- control-arm provenance resolver (5.)


def _make_provenance_item() -> Item:
    """A tiny, hand-verifiable item: the rival lacks date_of_birth: exactly one sibling
    ("Sibling Person") carries it (the only possible R1 source, so r1_examined=1 is
    unambiguous), and that same sibling is also the only R2 candidate once date_of_birth
    itself and the rival are excluded (a "child" sentence, so r2_examined=1 is unambiguous
    too, and its source happens to be a sibling *inside this item* -- exercising the
    same-item branch of the resolver, the opposite of what the real corpus shows)."""
    return Item(
        item_id="book-1",
        question="Who wrote the foreword?",
        answer="Gold Person",
        gold_title="Gold Person",
        options=[
            Entity("Gold Person", ["Gold Person is a person."]),
            Entity("Rival Person", ["Rival Person is a novelist."]),
            Entity(
                "Sibling Person",
                [
                    "Sibling Person was born in 1980.",
                    "Sibling Person's child is Kid Person.",
                ],
            ),
        ],
        relations=[],
    )


def _make_provenance_stage2_row(item_id="book-1", built=True, r1_examined=1, r2_examined=1):
    return {
        "model": "m1",
        "item_id": item_id,
        "rival_title": "Rival Person",
        "attribute": "date_of_birth",
        "built": built,
        "r1_examined": r1_examined,
        "r2_examined": r2_examined,
        "r1_total": 1,
        "r2_total": 1,
    }


def test_provenance_resolver_computes_rates_when_everything_resolves(tmp_path):
    from pilot.analyze_run3 import PARTS

    item = _make_provenance_item()
    corpus_index = build_corpus_index([item])
    items_by_id = {item.item_id: item}

    part_dir = tmp_path / "exp3z"
    part_dir.mkdir()
    PARTS["Z"] = ("exp3z", "test fixture")
    try:
        _write_jsonl(part_dir / "stage2_m1.jsonl", [_make_provenance_stage2_row()])
        result = resolve_control_provenance(
            tmp_path, "Z", items_by_id=items_by_id, corpus_index=corpus_index
        )
    finally:
        del PARTS["Z"]

    assert result["resolved"] == 1
    assert result["unresolved"] == 0
    assert result["unresolved_reasons"] == {}
    # The winning R2 source ("Sibling Person") is itself one of this item's own options.
    assert result["same_item"] == 1
    assert result["cross_item"] == 0
    assert result["cross_item_rate"] == 0.0
    assert result["relation_counts"] == {"child": 1}
    assert result["relation_rates"] == {"child": 1.0}
    assert result["same_relation_as_named"] == 0  # named date_of_birth, control is child
    assert result["r1_pool_median"] == 1
    assert result["r2_pool_median"] == 1


def test_provenance_resolver_refuses_a_percentage_when_a_row_is_unresolved(tmp_path):
    """One row resolves cleanly (see the previous test); a second row references an item id
    that is not in the lookup at all -- a coverage gap exactly like the one ITEM_LOOKUP_N_ITEMS
    exists to prevent in the real corpus. The resolver must still report resolved/unresolved
    counts, but cross_item_rate and relation_rates must come back None rather than a percentage
    computed over only the items that happened to resolve."""
    from pilot.analyze_run3 import PARTS

    item = _make_provenance_item()
    corpus_index = build_corpus_index([item])
    items_by_id = {item.item_id: item}  # "missing-item" deliberately absent

    part_dir = tmp_path / "exp3z"
    part_dir.mkdir()
    PARTS["Z"] = ("exp3z", "test fixture")
    try:
        rows = [
            _make_provenance_stage2_row(item_id="book-1"),
            _make_provenance_stage2_row(item_id="missing-item"),
        ]
        _write_jsonl(part_dir / "stage2_m1.jsonl", rows)
        result = resolve_control_provenance(
            tmp_path, "Z", items_by_id=items_by_id, corpus_index=corpus_index
        )
    finally:
        del PARTS["Z"]

    assert result["resolved"] == 1
    assert result["unresolved"] == 1
    assert result["unresolved_reasons"] == {"item_id_not_in_lookup": 1}
    assert result["cross_item_rate"] is None
    assert result["relation_rates"] is None


def test_provenance_resolver_refuses_a_percentage_on_an_out_of_range_examined_index(tmp_path):
    """r2_examined points past the end of the (correctly rebuilt, size-1) R2 candidate list --
    this must count as unresolved, not silently clamp or raise, and must still withhold the
    percentage."""
    from pilot.analyze_run3 import PARTS

    item = _make_provenance_item()
    corpus_index = build_corpus_index([item])
    items_by_id = {item.item_id: item}

    part_dir = tmp_path / "exp3z"
    part_dir.mkdir()
    PARTS["Z"] = ("exp3z", "test fixture")
    try:
        rows = [_make_provenance_stage2_row(r2_examined=7)]  # only 1 real candidate exists
        _write_jsonl(part_dir / "stage2_m1.jsonl", rows)
        result = resolve_control_provenance(
            tmp_path, "Z", items_by_id=items_by_id, corpus_index=corpus_index
        )
    finally:
        del PARTS["Z"]

    assert result["resolved"] == 0
    assert result["unresolved"] == 1
    assert result["unresolved_reasons"] == {"r2_examined_out_of_range": 1}
    assert result["cross_item_rate"] is None
    assert result["relation_rates"] is None


def test_provenance_resolver_ignores_rows_that_were_not_built():
    """A dropped (built: false) row must never enter the resolved/unresolved accounting at
    all -- it was never a candidate for repair, so it is neither a resolution nor a failure."""
    from pathlib import Path
    import tempfile

    from pilot.analyze_run3 import PARTS

    item = _make_provenance_item()
    corpus_index = build_corpus_index([item])
    items_by_id = {item.item_id: item}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        part_dir = tmp_path / "exp3z"
        part_dir.mkdir()
        PARTS["Z"] = ("exp3z", "test fixture")
        try:
            rows = [_make_provenance_stage2_row(built=False)]
            _write_jsonl(part_dir / "stage2_m1.jsonl", rows)
            result = resolve_control_provenance(
                tmp_path, "Z", items_by_id=items_by_id, corpus_index=corpus_index
            )
        finally:
            del PARTS["Z"]

    assert result["resolved"] == 0
    assert result["unresolved"] == 0
    assert result["cross_item_rate"] is None  # no resolved rows at all


# ------------------------------------------------------- fluency conditioning helpers (6.)


def _surp_row(model, part, item_id, condition, mean_nll, complete=True):
    return {
        "model": model, "part": part, "item_id": item_id, "condition": condition,
        "mean_nll": mean_nll, "complete": complete,
    }


def test_dedup_surprisal_rows_keeps_the_last_write_and_counts_the_rest():
    """Two writes for the same (model, part, item_id, condition) -- as a model re-run appending
    to its own file would leave -- must resolve to the *later* row (mean_nll=9.0, not the
    stale 1.0), with exactly one duplicate counted."""
    rows = [
        _surp_row("m1", "A", "i1", "R1", 1.0),
        _surp_row("m1", "A", "i2", "R1", 2.0),  # distinct key, not a duplicate
        _surp_row("m1", "A", "i1", "R1", 9.0),  # re-run of the first row's key
    ]
    out, n_dropped = _dedup_surprisal_rows(rows)
    assert n_dropped == 1
    assert len(out) == 2
    assert out[("m1", "A", "i1", "R1")]["mean_nll"] == 9.0
    assert out[("m1", "A", "i2", "R1")]["mean_nll"] == 2.0


def test_dedup_surprisal_rows_is_zero_when_every_key_is_unique():
    rows = [_surp_row("m1", "A", f"i{i}", "R1", float(i)) for i in range(5)]
    out, n_dropped = _dedup_surprisal_rows(rows)
    assert n_dropped == 0
    assert len(out) == 5


def test_load_fluency_choices_recomputes_from_the_raw_response(tmp_path):
    """``load_fluency_choices`` used to trust each stage-3 row's stored ``chosen_is_edited`` --
    written at generation time with the pre-fix parser -- rather than re-deriving it, so it
    stayed wrong for the fluency section even after ``analyze_run3.load_part`` was fixed. It
    must now re-derive ``choice``/``chosen_is_edited`` from the raw ``response`` the same way,
    via stage-1's own ``option_titles``."""
    from pilot.analyze_run3 import PARTS

    part_dir = tmp_path / "exp3z"
    part_dir.mkdir()
    PARTS["Z"] = ("exp3z", "test fixture")
    try:
        _write_jsonl(part_dir / "stage1_m1.jsonl",
                     [{"model": "m1", "item_id": "i1", "option_titles": ["Anna", "Bruno"]}])
        _write_jsonl(part_dir / "stage3_m1.jsonl", [
            # stored chosen_is_edited is wrong (written by the old parser); the raw response
            # plainly opens with "A) Anna", so the re-derived choice must be A, not B.
            {"model": "m1", "item_id": "i1", "condition": "R1", "response": "A) Anna",
             "edited_letter": "A", "chosen_is_edited": False},
            {"model": "m1", "item_id": "i1", "condition": "R2", "response": "B) Bruno",
             "edited_letter": "A", "chosen_is_edited": True},
        ])
        choices = load_fluency_choices(tmp_path, ["Z"])
    finally:
        del PARTS["Z"]

    assert choices[("m1", "Z", "i1", "R1")] is True
    assert choices[("m1", "Z", "i1", "R2")] is False


def test_load_fluency_choices_raises_on_missing_titles(tmp_path):
    from pilot.analyze_run3 import PARTS

    part_dir = tmp_path / "exp3z2"
    part_dir.mkdir()
    PARTS["Z2"] = ("exp3z2", "test fixture")
    try:
        _write_jsonl(part_dir / "stage3_m1.jsonl",
                     [{"model": "m1", "item_id": "i1", "condition": "R1", "response": "A"}])
        try:
            load_fluency_choices(tmp_path, ["Z2"])
            assert False, "expected a KeyError for the missing stage-1 titles"
        except KeyError:
            pass
    finally:
        del PARTS["Z2"]


def test_join_fluency_contrast_drops_rows_missing_either_condition_or_a_choice():
    """Four candidate items: one complete on both sides with both choices readable (kept), one
    missing its R2 surprisal row entirely, one with an R2 read marked incomplete, one with a
    null chosen_is_edited on R1 -- only the first survives the join."""
    surp = {
        ("m1", "A", "keep", "R1"): _surp_row("m1", "A", "keep", "R1", 1.0),
        ("m1", "A", "keep", "R2"): _surp_row("m1", "A", "keep", "R2", 2.0),
        ("m1", "A", "no-r2", "R1"): _surp_row("m1", "A", "no-r2", "R1", 1.0),
        ("m1", "A", "incomplete", "R1"): _surp_row("m1", "A", "incomplete", "R1", 1.0),
        ("m1", "A", "incomplete", "R2"): _surp_row("m1", "A", "incomplete", "R2", 2.0, complete=False),
        ("m1", "A", "no-choice", "R1"): _surp_row("m1", "A", "no-choice", "R1", 1.0),
        ("m1", "A", "no-choice", "R2"): _surp_row("m1", "A", "no-choice", "R2", 2.0),
    }
    choices = {
        ("m1", "A", "keep", "R1"): True, ("m1", "A", "keep", "R2"): False,
        ("m1", "A", "no-r2", "R1"): True,
        ("m1", "A", "incomplete", "R1"): True, ("m1", "A", "incomplete", "R2"): False,
        ("m1", "A", "no-choice", "R1"): None, ("m1", "A", "no-choice", "R2"): False,
    }
    joined = join_fluency_contrast(surp, choices, "R1", "R2")
    assert [row["item_id"] for row in joined] == ["keep"]
    assert joined[0]["delta_nll"] == 1.0 - 2.0
    assert joined[0]["chosen_a"] is True and joined[0]["chosen_b"] is False


def test_fluency_tercile_bounds_matches_the_released_881_row_split():
    """The real R3-R4 join in this project's own committed data has exactly 881 rows and splits
    293/294/294 -- the boundary rule ``floor(i*n/3)`` reproduces that split, not the more
    obvious "remainder to the first groups" one (which would give 294/294/293 instead)."""
    assert fluency_tercile_bounds(881) == [0, 293, 587, 881]


def test_fluency_tercile_bounds_on_a_small_hand_case():
    assert fluency_tercile_bounds(9) == [0, 3, 6, 9]
    assert fluency_tercile_bounds(10) == [0, 3, 6, 10]  # remainder goes to the last tercile
    assert fluency_tercile_bounds(11) == [0, 3, 7, 11]  # remainder goes to the last two


def _joined_row(key, delta, chosen_a=True, chosen_b=False):
    model, part, item_id = key
    return {
        "key": key, "model": model, "part": part, "item_id": item_id,
        "mean_nll_a": 0.0, "mean_nll_b": -delta, "delta_nll": delta,
        "chosen_a": chosen_a, "chosen_b": chosen_b,
    }


def test_fluency_terciles_splits_a_small_hand_sorted_set_correctly():
    """Nine rows with distinct, already-ascending deltas 0..8 -- three equal terciles of three,
    each tercile's rows in ascending-delta order."""
    rows = [_joined_row(("m1", "A", f"i{i}"), float(i)) for i in range(9)]
    # shuffle the input order so the function's own sort, not accidental input order, is tested
    shuffled = [rows[i] for i in (5, 0, 8, 3, 1, 7, 2, 6, 4)]
    terciles = fluency_terciles(shuffled)
    assert [len(t) for t in terciles] == [3, 3, 3]
    assert [row["delta_nll"] for row in terciles[0]] == [0.0, 1.0, 2.0]
    assert [row["delta_nll"] for row in terciles[1]] == [3.0, 4.0, 5.0]
    assert [row["delta_nll"] for row in terciles[2]] == [6.0, 7.0, 8.0]


def test_fluency_terciles_ties_break_deterministically_on_the_row_key():
    """Two rows share the same delta -- the tie must break on ``key`` (alphabetic), the same
    order ``join_fluency_contrast`` itself produces, so a rerun never reshuffles which tercile a
    tied item lands in."""
    rows = [
        _joined_row(("m1", "A", "b"), 1.0),
        _joined_row(("m1", "A", "a"), 1.0),
        _joined_row(("m1", "A", "c"), 2.0),
    ]
    terciles = fluency_terciles(rows)
    ordered_ids = [row["item_id"] for t in terciles for row in t]
    assert ordered_ids == ["a", "b", "c"]


def test_fluency_loo_within_tercile_drops_only_the_named_models_rows():
    """Three rows from two models -- dropping "m1" must leave exactly the "m2" rows behind, and
    dropping "m2" must leave exactly the "m1" rows, checked by feeding
    ``fluency_tercile_discrete`` a population engineered so b/c differ once a model is gone."""
    rows = [
        _joined_row(("m1", "A", "i1"), -1.0, chosen_a=True, chosen_b=False),  # m1: b
        _joined_row(("m1", "A", "i2"), -0.5, chosen_a=True, chosen_b=False),  # m1: b
        _joined_row(("m2", "A", "i3"), 0.5, chosen_a=False, chosen_b=True),   # m2: c
    ]
    loo = fluency_loo_within_tercile(rows, "R1", "R2")
    by_dropped = {row["dropped"]: row for row in loo}

    assert set(by_dropped) == {None, "m1", "m2"}
    assert by_dropped[None]["discrete"]["b_a_only"] == 2
    assert by_dropped[None]["discrete"]["c_b_only"] == 1

    # dropping m1 leaves only m2's single discordant "c" row
    assert by_dropped["m1"]["discrete"]["b_a_only"] == 0
    assert by_dropped["m1"]["discrete"]["c_b_only"] == 1
    assert by_dropped["m1"]["n_models"] == 1

    # dropping m2 leaves only m1's two discordant "b" rows
    assert by_dropped["m2"]["discrete"]["b_a_only"] == 2
    assert by_dropped["m2"]["discrete"]["c_b_only"] == 0
    assert by_dropped["m2"]["n_models"] == 1


def test_fluency_tercile_discrete_matches_a_hand_computed_odds_ratio():
    rows = [
        _joined_row(("m1", "A", "i1"), -1.0, chosen_a=True, chosen_b=False),
        _joined_row(("m1", "A", "i2"), -1.0, chosen_a=True, chosen_b=False),
        _joined_row(("m1", "A", "i3"), -1.0, chosen_a=False, chosen_b=True),
        _joined_row(("m1", "A", "i4"), -1.0, chosen_a=True, chosen_b=True),  # concordant
    ]
    d = fluency_tercile_discrete(rows, "R1", "R2")
    assert d["b_a_only"] == 2
    assert d["c_b_only"] == 1
    assert math.isclose(d["or"]["or"], 2.0, rel_tol=1e-12)


# --------------------------------------------------------------- no stray control characters


def test_touched_files_contain_no_stray_control_characters():
    """A patch once wrote a literal backspace where a regex needed a word boundary -- see
    test_repair.py / test_extract.py / test_analyze_run3.py's identical guard, scoped here to
    the files this task touched."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    paths = [here.parent / "pilot" / "analyze_round5.py", here / "test_analyze_round5.py"]

    offenders = []
    for path in paths:
        for i, byte in enumerate(path.read_bytes()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((str(path), i, byte))
    assert offenders == []
