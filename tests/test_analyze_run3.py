"""Tests for harness.analyze_run3: recomputes run-3 statistics from stage-3 JSONL fixtures.

Hand-written fixtures only; no GPU or network.
"""

from __future__ import annotations

import json
import math

from harness.analyze_run3 import (
    _margin,
    _median,
    _sign_counts,
    _trimmed_mean,
    baseline_by_target_paper,
    baseline_by_target_strict,
    bootstrap_stat,
    discordant_concordant_diffs,
    discrete_outcomes,
    gate_eight_skew,
    holm,
    load_part,
    margin_outcomes,
    odds_ratio,
    paired_diffs,
    paired_mean,
)


# ------------------------------------------------------------------------------------- _margin


def test_margin_of_a_winning_letter_is_positive():
    assert _margin({"A": 0.6, "B": 0.3, "C": 0.1}, "A") == 0.6 - 0.3


def test_margin_of_a_losing_letter_is_negative():
    assert _margin({"A": 0.2, "B": 0.5, "C": 0.3}, "A") == 0.2 - 0.5


def test_margin_of_an_exact_tie_with_the_best_competitor_is_zero():
    """A tie is the boundary an argmax has to break somehow; the margin itself must read
    exactly zero rather than a small nonzero value from float drift."""
    assert _margin({"A": 0.5, "B": 0.5, "C": 0.0}, "A") == 0.0


def test_margin_ignores_the_letter_itself_when_picking_the_best_competitor():
    """The competitor must be the best of the *other* letters, not accidentally itself again --
    a bug that would make every margin come out as zero."""
    assert _margin({"A": 0.9, "B": 0.05, "C": 0.05}, "A") == 0.9 - 0.05


def test_margin_is_none_when_the_letter_is_missing():
    assert _margin({"B": 0.6, "C": 0.4}, "A") is None


def test_margin_is_none_when_the_letter_is_the_only_one():
    """No "other" candidates to compare against -- undefined, not a fabricated zero."""
    assert _margin({"A": 1.0}, "A") is None


# ------------------------------------------------------------------------------ margin_outcomes


def _row(condition, *, edited_letter, probs, complete=True, chosen_is_edited=None):
    return {
        "condition": condition, "edited_letter": edited_letter,
        "chosen_is_edited": chosen_is_edited,
        "letter_probe": {"complete": complete, "probs": probs},
    }


def test_margin_outcomes_joins_each_condition_against_its_own_items_r0():
    items = {
        "item-1": {
            "R0": _row("R0", edited_letter=None, probs={"A": 0.5, "B": 0.5}),
            "R1": _row("R1", edited_letter="A", probs={"A": 0.7, "B": 0.3},
                       chosen_is_edited=True),
        },
    }
    out = margin_outcomes(items)

    assert set(out) == {"item-1"}
    rec = out["item-1"]["R1"]
    # before: A's margin at R0 = 0.5-0.5 = 0.0; after: 0.7-0.3 = 0.4; delta = 0.4
    assert math.isclose(rec["margin_before"], 0.0, abs_tol=1e-12)
    assert math.isclose(rec["margin_after"], 0.4, abs_tol=1e-12)
    assert math.isclose(rec["delta_margin"], 0.4, abs_tol=1e-12)
    assert rec["chosen_is_edited"] is True
    assert rec["complete"] is True


def test_margin_outcomes_excludes_an_item_whose_r0_probe_is_incomplete():
    items = {
        "item-1": {
            "R0": _row("R0", edited_letter=None, probs={"A": 0.5}, complete=False),
            "R1": _row("R1", edited_letter="A", probs={"A": 0.7, "B": 0.3}),
        },
    }
    assert margin_outcomes(items) == {}


def test_margin_outcomes_excludes_a_condition_whose_own_probe_is_incomplete():
    """R0 is fine; R1's own probe is not -- only R1 drops, not the whole item, and if nothing
    is left for that item it must not appear at all."""
    items = {
        "item-1": {
            "R0": _row("R0", edited_letter=None, probs={"A": 0.5, "B": 0.5}),
            "R1": _row("R1", edited_letter="A", probs={"A": 0.7}, complete=False),
            "R2": _row("R2", edited_letter="A", probs={"A": 0.4, "B": 0.6}),
        },
    }
    out = margin_outcomes(items)
    assert set(out["item-1"]) == {"R2"}


def test_margin_outcomes_excludes_a_condition_with_no_edited_letter():
    items = {
        "item-1": {
            "R0": _row("R0", edited_letter=None, probs={"A": 0.5, "B": 0.5}),
            "R1": _row("R1", edited_letter=None, probs={"A": 0.7, "B": 0.3}),
        },
    }
    assert margin_outcomes(items) == {}


def test_margin_outcomes_never_reads_r0_against_itself():
    """R0 carries no `edited_letter`, so it must never produce its own margin record."""
    items = {
        "item-1": {
            "R0": _row("R0", edited_letter=None, probs={"A": 0.5, "B": 0.5}),
        },
    }
    assert margin_outcomes(items) == {}


# ----------------------------------------------------------------------- paired_diffs / paired_mean


def test_paired_diffs_only_uses_items_complete_on_both_sides():
    outcomes = {
        "i1": {"R2": {"delta_margin": 0.3}, "R4": {"delta_margin": 0.1}},
        "i2": {"R2": {"delta_margin": 0.2}},  # R4 missing entirely
        "i3": {"R2": {"delta_margin": None}, "R4": {"delta_margin": 0.05}},  # R2 unreadable
        "i4": {"R2": {"delta_margin": -0.1}, "R4": {"delta_margin": 0.4}},
    }
    diffs = paired_diffs(outcomes, "R2", "R4", "delta_margin")
    assert sorted(diffs) == sorted([0.3 - 0.1, -0.1 - 0.4])


def test_paired_mean_matches_a_hand_computed_mean():
    outcomes = {
        "i1": {"R2": {"delta_margin": 0.3}, "R4": {"delta_margin": 0.1}},
        "i2": {"R2": {"delta_margin": -0.1}, "R4": {"delta_margin": 0.4}},
    }
    result = paired_mean(outcomes, "R2", "R4", "delta_margin", seed=1)
    assert math.isclose(result["mean"], ((0.3 - 0.1) + (-0.1 - 0.4)) / 2, rel_tol=1e-12)
    assert result["contrast"] == "R2-R4"


# --------------------------------------------------------------------------------------- holm


def test_holm_on_a_hand_worked_five_test_family():
    """m=5. Sorted p: 0.005, 0.010, 0.020, 0.030, 0.050.
    Step-down multipliers (m-i) for i=0..4: 5, 4, 3, 2, 1.
    Running max of (multiplier * p), monotone non-decreasing:
      0.005*5=0.025            -> 0.025
      0.010*4=0.040, max(.025,.040)=0.040
      0.020*3=0.060, max(.040,.060)=0.060
      0.030*2=0.060, max(.060,.060)=0.060
      0.050*1=0.050, max(.060,.050)=0.060
    """
    raw = {"a": 0.005, "b": 0.010, "c": 0.020, "d": 0.030, "e": 0.050}
    adj = holm(raw)
    expected = {"a": 0.025, "b": 0.040, "c": 0.060, "d": 0.060, "e": 0.060}
    for key, value in expected.items():
        assert math.isclose(adj[key], value, rel_tol=1e-12), (key, adj[key], value)


def test_holm_is_monotone_and_capped_at_one():
    raw = {"a": 0.6, "b": 0.7, "c": 0.9}
    adj = holm(raw)
    ordered = sorted(adj.items(), key=lambda kv: raw[kv[0]])
    values = [v for _, v in ordered]
    assert values == sorted(values)  # non-decreasing in p-value order
    assert all(v <= 1.0 for v in values)


# ----------------------------------------------------------------------------------- odds_ratio


def test_odds_ratio_with_a_zero_cell_applies_haldane_anscombe():
    o = odds_ratio(0, 12)
    assert o["haldane_anscombe"] is True
    # (0+0.5)/(12+0.5) = 0.5/12.5
    assert math.isclose(o["or"], 0.5 / 12.5, rel_tol=1e-9)
    assert o["lo"] < o["or"] < o["hi"]


def test_odds_ratio_with_both_cells_zero_gives_one_and_is_still_corrected():
    o = odds_ratio(0, 0)
    assert o["haldane_anscombe"] is True
    assert math.isclose(o["or"], 1.0, rel_tol=1e-9)


def test_odds_ratio_without_a_zero_cell_is_uncorrected():
    o = odds_ratio(10, 5)
    assert o["haldane_anscombe"] is False
    assert math.isclose(o["or"], 2.0, rel_tol=1e-9)


# ------------------------------------------------------------------ sign / median / trimmed mean


def test_sign_counts_hand_example():
    assert _sign_counts([0.1, -0.2, 0.0, 0.3, -0.05, 0.0]) == (2, 2, 2)


def test_median_odd_and_even_length():
    assert _median([5.0, 1.0, 3.0]) == 3.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_of_empty_is_none():
    assert _median([]) is None


def test_trimmed_mean_drops_symmetric_tails():
    # ten values 0..9, 10% trim drops the smallest and the largest one each.
    values = [float(i) for i in range(10)]
    assert math.isclose(_trimmed_mean(values, 0.10), sum(range(1, 9)) / 8, rel_tol=1e-12)


def test_trimmed_mean_falls_back_to_plain_mean_on_a_tiny_sample():
    values = [1.0, 2.0, 3.0]
    # 40% trim of 3 items would drop everything (k=1 each side, 3-2=1 remains) -- still valid
    assert _trimmed_mean(values, 0.40) == 2.0


def test_trimmed_mean_of_empty_is_none():
    assert _trimmed_mean([], 0.05) is None


def test_bootstrap_stat_is_deterministic_under_a_fixed_seed():
    values = [0.1, 0.3, -0.05, 0.2, 0.15, 0.0, 0.25, -0.1, 0.05, 0.4]
    first = bootstrap_stat(values, _median, seed=99, n_resamples=500)
    second = bootstrap_stat(values, _median, seed=99, n_resamples=500)
    assert first == second


def test_bootstrap_stat_brackets_a_known_median():
    values = [float(i) for i in range(11)]  # median exactly 5.0
    result = bootstrap_stat(values, _median, seed=42, n_resamples=2000)
    assert result["stat"] == 5.0
    assert result["ci_low"] <= 5.0 <= result["ci_high"]


def test_bootstrap_stat_on_no_data_is_all_none():
    result = bootstrap_stat([], _median, seed=1)
    assert result == {"stat": None, "ci_low": None, "ci_high": None, "n": 0, "seed": 1}


def test_sign_counts_agree_with_median_direction_on_a_skewed_sample():
    """The scenario the paper's third explanation predicts: many small negative moves and one
    huge positive outlier. The mean is dragged positive by the outlier; the sign count and the
    median both stay negative -- exactly the decomposition this module is built to detect."""
    values = [-0.01, -0.02, -0.01, -0.015, -0.02, 5.0]
    pos, neg, zero = _sign_counts(values)
    assert (pos, neg, zero) == (1, 5, 0)
    assert _median(values) < 0
    assert sum(values) / len(values) > 0  # the mean disagrees with both


# --------------------------------------------------------------------- gate_eight_skew: the join


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_gate_eight_skew_excludes_unparseable_stage1_choices_from_the_denominator(tmp_path):
    """This is the exact defect task 2 found: `choice_correct` is `None` (not `False`) when
    stage 1's free-text answer could not be parsed at all, and a stage-2 row built on top of
    such an item must not silently count as an "incorrect attempt" -- it must not count as an
    attempt at all."""
    results = tmp_path
    part_dir = results / "exp3z"
    part_dir.mkdir()
    from harness.analyze_run3 import PARTS
    PARTS["Z"] = ("exp3z", "test fixture")
    try:
        # choice_correct is no longer trusted as written -- gate_eight_skew re-derives it from
        # each row's own response/option_titles/gold_letter with the fixed parser, so the
        # fixture supplies those instead of the stale field directly.
        stage1 = [
            {"model": "m1", "item_id": "ok-correct", "gold_letter": "A",
             "option_titles": ["Foo", "Bar"], "response": "A) Foo"},
            {"model": "m1", "item_id": "ok-wrong", "gold_letter": "A",
             "option_titles": ["Foo", "Bar"], "response": "B) Bar"},
            {"model": "m1", "item_id": "unparseable", "gold_letter": "A",
             "option_titles": ["Foo", "Bar"], "response": "I have no idea which one."},
        ]
        stage2 = [
            {"model": "m1", "item_id": "ok-correct", "built": True},
            {"model": "m1", "item_id": "ok-wrong", "built": False},
            {"model": "m1", "item_id": "unparseable", "built": True},
            {"model": "m1", "item_id": "no-stage1-row", "built": False},
        ]
        _write_jsonl(part_dir / "stage1_m1.jsonl", stage1)
        _write_jsonl(part_dir / "stage2_m1.jsonl", stage2)

        g = gate_eight_skew(results, "Z")

        # attempted pool: only the two rows with a real stage-1 correctness signal.
        assert g["attempted_n"] == 2
        assert g["attempted_correct"] == 1
        # built pool: "ok-correct" (built, correct) and "unparseable" is excluded entirely, so
        # it must not appear in built_n either even though its stage-2 row says built=True.
        assert g["built_n"] == 1
        assert g["built_correct"] == 1
        assert g["unjoined_unparseable_choice"] == 1
        assert g["unjoined_missing_key"] == 1
        assert g["unjoined"] == 2
    finally:
        del PARTS["Z"]


# --------------------------------------------------------- baseline_by_target: paper vs. strict


def _r0_row(probs, complete=True):
    return {"condition": "R0", "letter_probe": {"complete": complete, "probs": probs}}


def _target_row(condition, edited_letter, r0_p_target, complete=True):
    return {"condition": condition, "edited_letter": edited_letter,
            "r0_p_target": r0_p_target,
            "letter_probe": {"complete": complete}}


def test_baseline_by_target_paper_reads_r0_directly_even_if_the_rows_own_probe_is_incomplete():
    """This is the reconciliation from task 1: the paper's Table 3 population only requires R0's
    own probe to contain the letter, not the R1/R3 row's own probe to be complete."""
    items = {
        "item-1": {
            "R0": _r0_row({"A": 0.2, "B": 0.5, "C": 0.3}),
            # R1's own probe is incomplete -- r0_p_target was therefore never set by the run --
            # but the paper population must still pick up R0's own read of "A".
            "R1": _target_row("R1", "A", None, complete=False),
            "R3": _target_row("R3", "C", None, complete=False),
        },
    }
    paper = baseline_by_target_paper(items)
    assert paper["rival"]["n"] == 1
    assert math.isclose(paper["rival"]["mean"], 0.2, rel_tol=1e-12)
    assert paper["third"]["n"] == 1
    assert math.isclose(paper["third"]["mean"], 0.3, rel_tol=1e-12)

    strict = baseline_by_target_strict(items)
    assert strict["rival"]["n"] == 0
    assert strict["rival"]["mean"] is None


def test_baseline_by_target_strict_only_uses_the_r0_p_target_field():
    items = {
        "item-1": {
            "R0": _r0_row({"A": 0.9, "B": 0.1}),  # ignored by the strict reader
            "R1": _target_row("R1", "A", 0.4),
            "R3": _target_row("R3", "B", 0.1),
        },
    }
    strict = baseline_by_target_strict(items)
    assert strict["rival"]["n"] == 1
    assert math.isclose(strict["rival"]["mean"], 0.4, rel_tol=1e-12)
    assert strict["third"]["n"] == 1
    assert math.isclose(strict["third"]["mean"], 0.1, rel_tol=1e-12)


def test_baseline_by_target_paper_and_strict_agree_when_every_probe_is_complete():
    """Part B's case: when nothing is ever partially complete, the two populations coincide."""
    items = {
        "item-1": {
            "R0": _r0_row({"A": 0.4, "B": 0.6}),
            "R1": _target_row("R1", "A", 0.4),
            "R3": _target_row("R3", "B", 0.6),
        },
    }
    paper = baseline_by_target_paper(items)
    strict = baseline_by_target_strict(items)
    assert paper["rival"] == strict["rival"]
    assert paper["third"] == strict["third"]


def test_baseline_by_target_ratio_is_none_without_data():
    out = baseline_by_target_paper({})
    assert out["rival"] == {"n": 0, "mean": None}
    assert out["ratio"] is None


# -------------------------------------------------------------------------------- load_part


def test_load_part_keys_by_model_then_item_then_condition(tmp_path, monkeypatch):
    from harness.analyze_run3 import PARTS

    part_dir = tmp_path / "exp3y"
    part_dir.mkdir()
    PARTS["Y"] = ("exp3y", "test fixture")
    try:
        # load_part now needs option_titles from this part's own stage 1 (part != "A") for
        # every (model, item_id) it loads -- see test_load_part_raises_on_missing_titles below
        # for what happens without this.
        _write_jsonl(part_dir / "stage1_m1.jsonl",
                     [{"model": "m1", "item_id": "i1", "option_titles": ["Anna", "Bruno"]}])
        _write_jsonl(part_dir / "stage1_m2.jsonl",
                     [{"model": "m2", "item_id": "i1", "option_titles": ["Anna", "Bruno"]}])
        rows = [
            {"model": "m1", "item_id": "i1", "condition": "R0", "x": 1, "response": "A"},
            {"model": "m1", "item_id": "i1", "condition": "R1", "x": 2, "response": "A"},
            {"model": "m2", "item_id": "i1", "condition": "R0", "x": 3, "response": "A"},
        ]
        _write_jsonl(part_dir / "stage3_m1.jsonl",
                     [r for r in rows if r["model"] == "m1"])
        _write_jsonl(part_dir / "stage3_m2.jsonl",
                     [r for r in rows if r["model"] == "m2"])

        loaded = load_part(tmp_path, "Y")
        assert set(loaded) == {"m1", "m2"}
        assert loaded["m1"]["i1"]["R0"]["x"] == 1
        assert loaded["m1"]["i1"]["R1"]["x"] == 2
        # same item_id under a different model must not collide with m1's record.
        assert loaded["m2"]["i1"]["R0"]["x"] == 3
    finally:
        del PARTS["Y"]


def test_load_part_raises_on_missing_titles(tmp_path):
    """A stage-3 row whose (model, item_id) has no stage-1 titles is a coverage hole and must
    raise, not be silently skipped -- this is the loud-failure task 1 requires."""
    from harness.analyze_run3 import PARTS

    part_dir = tmp_path / "exp3z"
    part_dir.mkdir()
    PARTS["Z2"] = ("exp3z", "test fixture")
    try:
        # no stage1_*.jsonl written at all -- titles map is empty.
        _write_jsonl(part_dir / "stage3_m1.jsonl",
                     [{"model": "m1", "item_id": "i1", "condition": "R0", "response": "A"}])
        try:
            load_part(tmp_path, "Z2")
            assert False, "expected a KeyError for the missing stage-1 titles"
        except KeyError:
            pass
    finally:
        del PARTS["Z2"]


# --------------------------------------------------------- discordant / concordant split


def test_discordant_concordant_splits_on_the_discrete_outcome_not_the_continuous_one():
    """The split key is `chosen_is_edited` differing between R2 and R4 -- exactly what
    McNemar's b/c are built from -- independent of what the continuous value itself is."""
    disc = {
        "flip": {"R2": True, "R4": False},   # discordant: McNemar would count this
        "flip2": {"R2": False, "R4": True},  # discordant, the other direction
        "same-true": {"R2": True, "R4": True},    # concordant
        "same-false": {"R2": False, "R4": False},  # concordant
    }
    other = {
        "flip": {"R2": {"delta_margin": 0.3}, "R4": {"delta_margin": -0.1}},
        "flip2": {"R2": {"delta_margin": 0.1}, "R4": {"delta_margin": 0.2}},
        "same-true": {"R2": {"delta_margin": 0.05}, "R4": {"delta_margin": 0.02}},
        "same-false": {"R2": {"delta_margin": -0.2}, "R4": {"delta_margin": -0.4}},
    }
    discordant, concordant = discordant_concordant_diffs(disc, other, "R2", "R4", "delta_margin")

    assert sorted(discordant) == sorted([0.3 - (-0.1), 0.1 - 0.2])
    assert sorted(concordant) == sorted([0.05 - 0.02, -0.2 - (-0.4)])


def test_discordant_concordant_excludes_items_missing_a_discrete_read():
    disc = {
        "no-r2-read": {"R2": None, "R4": True},  # discrete read on R2 itself unreadable
        # "not-in-disc" is deliberately absent here -- no discrete row for it at all.
    }
    other = {
        "no-r2-read": {"R2": {"delta_margin": 0.1}, "R4": {"delta_margin": 0.2}},
        "not-in-disc": {"R2": {"delta_margin": 0.5}, "R4": {"delta_margin": 0.1}},
    }
    discordant, concordant = discordant_concordant_diffs(disc, other, "R2", "R4", "delta_margin")
    assert discordant == []
    assert concordant == []


def test_discordant_concordant_excludes_items_missing_the_continuous_field():
    disc = {"item-1": {"R2": True, "R4": False}}
    other = {"item-1": {"R2": {"delta_margin": 0.1}, "R4": {"delta_margin": None}}}
    discordant, concordant = discordant_concordant_diffs(disc, other, "R2", "R4", "delta_margin")
    assert discordant == []
    assert concordant == []


def test_discordant_concordant_totals_match_a_plain_paired_diff_partition():
    """Every item that appears in the plain `paired_diffs` population must land in exactly one
    of the two subsets when a full discrete read is available -- no item silently vanishes or
    is double counted."""
    disc = {
        "i1": {"R2": True, "R4": False},
        "i2": {"R2": True, "R4": True},
        "i3": {"R2": False, "R4": True},
        "i4": {"R2": False, "R4": False},
    }
    other = {iid: {"R2": {"delta_margin": 0.1 * n}, "R4": {"delta_margin": 0.05 * n}}
              for n, iid in enumerate(disc, start=1)}
    discordant, concordant = discordant_concordant_diffs(disc, other, "R2", "R4", "delta_margin")
    all_diffs = paired_diffs(other, "R2", "R4", "delta_margin")
    assert sorted(discordant + concordant) == sorted(all_diffs)
    assert len(discordant) == 2  # i1, i3
    assert len(concordant) == 2  # i2, i4


def test_discordant_concordant_can_disagree_in_direction_from_each_other():
    """The scenario this split exists to detect: the discordant subset moves one way and the
    far larger concordant subset moves the other way, so a pooled mean can point opposite to
    McNemar's own direction without either measure being broken."""
    disc = {f"d{i}": {"R2": True, "R4": False} for i in range(3)}
    disc.update({f"c{i}": {"R2": True, "R4": True} for i in range(40)})
    other = {f"d{i}": {"R2": {"delta_margin": 0.5}, "R4": {"delta_margin": 0.0}}
              for i in range(3)}  # discordant: strongly favours R2 (+0.5 each)
    other.update({f"c{i}": {"R2": {"delta_margin": -0.05}, "R4": {"delta_margin": 0.0}}
                   for i in range(40)})  # concordant: mildly favours R4, but there are many
    discordant, concordant = discordant_concordant_diffs(disc, other, "R2", "R4", "delta_margin")

    assert all(v > 0 for v in discordant)   # discordant subset unanimously favours R2
    assert all(v < 0 for v in concordant)   # concordant subset unanimously favours R4
    pooled_mean = sum(discordant + concordant) / len(discordant + concordant)
    assert pooled_mean < 0  # the numerous concordant items dominate the pooled mean


def test_discordant_concordant_uses_discrete_outcomes_shape_directly():
    """Sanity check against `discrete_outcomes` itself, not a hand-rolled stand-in, so a future
    change to that function's output shape is caught here too."""
    items = {
        "i1": {"R2": {"chosen_is_edited": True}, "R4": {"chosen_is_edited": False}},
        "i2": {"R2": {"chosen_is_edited": True}, "R4": {"chosen_is_edited": True}},
    }
    disc = discrete_outcomes(items)
    other = {
        "i1": {"R2": {"delta_margin": 0.2}, "R4": {"delta_margin": 0.1}},
        "i2": {"R2": {"delta_margin": 0.3}, "R4": {"delta_margin": 0.1}},
    }
    discordant, concordant = discordant_concordant_diffs(disc, other, "R2", "R4", "delta_margin")
    assert discordant == [0.2 - 0.1]
    assert concordant == [0.3 - 0.1]


# --------------------------------------------------------------- no stray control characters


def test_touched_files_contain_no_stray_control_characters():
    """A patch once wrote a literal backspace where a regex needed a word boundary -- see
    test_repair.py / test_extract.py / test_r3_recency.py's identical guard, scoped here to the
    files this task touched."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    paths = [here.parent / "harness" / "analyze_run3.py", here / "test_analyze_run3.py"]

    offenders = []
    for path in paths:
        for i, byte in enumerate(path.read_bytes()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((str(path), i, byte))
    assert offenders == []
