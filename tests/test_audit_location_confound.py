"""Unit tests for the offline third-option-baseline confound audit."""

from harness.audit_location_confound import (
    BAND_FLOOR,
    LOOSE_BAND,
    TIGHT_BAND,
    TOO_SMALL_TO_BIN,
    band_membership,
    dose_response_series,
    items_in_band,
    location_baselines,
    pearson_r,
    permutation_corr_test,
    quantile_bin_means,
    summarize_baseline_pairs,
)


def _r0(probs):
    return {"letter_probe": {"probs": probs}}


# ------------------------------------------------------------------------------- section 1


def test_location_baselines_pairs_rival_and_third_from_r0_probs():
    items = {
        "i1": {
            "R0": _r0({"A": 0.5, "B": 0.1, "C": 0.4}),
            "R1": {"edited_letter": "A"},
            "R3": {"edited_letter": "C"},
        },
        # Missing R3: excluded entirely, not partially included.
        "i2": {
            "R0": _r0({"A": 0.5, "B": 0.5}),
            "R1": {"edited_letter": "A"},
        },
        # R3's letter not in R0's probs: excluded.
        "i3": {
            "R0": _r0({"A": 0.5}),
            "R1": {"edited_letter": "A"},
            "R3": {"edited_letter": "Z"},
        },
        # No R0 at all: excluded.
        "i4": {"R1": {"edited_letter": "A"}, "R3": {"edited_letter": "B"}},
    }
    out = location_baselines(items)
    assert out == {"i1": (0.5, 0.4)}


def test_summarize_baseline_pairs_counts_below_and_paired_diff():
    pairs = {"i1": (0.5, 0.1), "i2": (0.2, 0.2), "i3": (0.1, 0.3)}
    s = summarize_baseline_pairs(pairs)
    assert s["n"] == 3
    assert abs(s["mean_rival"] - (0.5 + 0.2 + 0.1) / 3) < 1e-12
    assert abs(s["mean_third"] - (0.1 + 0.2 + 0.3) / 3) < 1e-12
    # i1: third(0.1) < rival(0.5) -> below. i2: equal -> not below. i3: third(0.3) > rival(0.1).
    assert s["below"] == 1
    assert abs(s["share_below"] - 1 / 3) < 1e-12
    diffs = [0.5 - 0.1, 0.2 - 0.2, 0.1 - 0.3]
    assert abs(s["diff_ci"]["mean"] - sum(diffs) / 3) < 1e-12


def test_summarize_baseline_pairs_handles_empty_input():
    s = summarize_baseline_pairs({})
    assert s == {"n": 0, "mean_rival": None, "mean_third": None, "diff_ci": None,
                 "below": 0, "share_below": None}


# ------------------------------------------------------------------------------- sections 2/3


def test_dose_response_series_discrete_reads_baseline_and_flip():
    items = {
        "i1": {
            "R3": {"r0_p_target": 0.2, "chosen_is_edited": False},
            "R4": {"chosen_is_edited": True},
        },
        # Missing baseline: excluded.
        "i2": {
            "R3": {"r0_p_target": None, "chosen_is_edited": True},
            "R4": {"chosen_is_edited": False},
        },
        # Missing R4: excluded.
        "i3": {"R3": {"r0_p_target": 0.3, "chosen_is_edited": True}},
    }
    rows = dose_response_series(items, "R3", "R4", "discrete")
    assert rows == [("i1", 0.2, -1.0)]  # False(R3)=0 - True(R4)=1 -> -1.0


def test_dose_response_series_continuous_reads_delta_p_edited():
    items = {
        "i1": {
            "R3": {"r0_p_target": 0.2, "delta_p_edited": 0.10},
            "R4": {"delta_p_edited": 0.02},
        },
        "i2": {
            "R3": {"r0_p_target": 0.4, "delta_p_edited": None},
            "R4": {"delta_p_edited": 0.02},
        },
    }
    rows = dose_response_series(items, "R3", "R4", "continuous")
    assert len(rows) == 1
    item_id, baseline, outcome = rows[0]
    assert item_id == "i1"
    assert baseline == 0.2
    assert abs(outcome - 0.08) < 1e-12


# ------------------------------------------------------------------------------- correlation


def test_pearson_r_perfect_positive_and_negative():
    assert abs(pearson_r([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(pearson_r([1, 2, 3, 4], [40, 30, 20, 10]) - (-1.0)) < 1e-9


def test_pearson_r_none_when_no_variance_or_too_few_points():
    assert pearson_r([1], [1]) is None
    assert pearson_r([1, 1, 1], [1, 2, 3]) is None  # zero variance on x
    assert pearson_r([], []) is None


def test_permutation_corr_test_is_deterministic_and_extreme_for_perfect_correlation():
    xs = [float(i) for i in range(10)]
    ys = [float(i) * 2 for i in range(10)]
    first = permutation_corr_test(xs, ys, seed=7, resamples=500)
    second = permutation_corr_test(xs, ys, seed=7, resamples=500)
    assert first == second
    assert first["n"] == 10
    assert abs(first["r"] - 1.0) < 1e-9
    # No reshuffle of ys can be as extreme as the perfectly-sorted pairing (barring the identity
    # permutation itself), so this should sit at the smallest p-value the resample count allows.
    assert first["p_value"] <= 2 / 501


def test_permutation_corr_test_handles_undefined_correlation():
    result = permutation_corr_test([1, 1, 1], [1, 2, 3])
    assert result["r"] is None
    assert result["p_value"] is None


# ------------------------------------------------------------------------------- binning


def test_quantile_bin_means_splits_sorted_data_and_reports_range():
    xs = [float(i) for i in range(20)]
    ys = [float(i) for i in range(20)]  # outcome == baseline: perfectly monotonic
    bins = quantile_bin_means(xs, ys, n_bins=4)
    assert len(bins) == 4
    assert sum(b["n"] for b in bins) == 20
    # Monotonic input must produce monotonically increasing bin means, both sides.
    means_x = [b["mean_baseline"] for b in bins]
    means_y = [b["mean_outcome"] for b in bins]
    assert means_x == sorted(means_x)
    assert means_y == sorted(means_y)
    assert bins[0]["baseline_lo"] == 0.0
    assert bins[-1]["baseline_hi"] == 19.0


def test_quantile_bin_means_too_small_returns_empty():
    xs = [1.0, 2.0, 3.0]
    ys = [1.0, 2.0, 3.0]
    assert len(xs) < TOO_SMALL_TO_BIN
    assert quantile_bin_means(xs, ys) == []


def test_quantile_bin_means_shrinks_bin_count_for_small_n():
    # 8 points, requested 4 bins: with the "no bin under ~5" rule, bins shrink to 1.
    xs = [float(i) for i in range(8)]
    ys = [float(i) for i in range(8)]
    bins = quantile_bin_means(xs, ys, n_bins=4)
    assert len(bins) == 1
    assert bins[0]["n"] == 8


# ------------------------------------------------------------------------------- section 4 bands


def test_band_membership_boundaries():
    # ratio exactly at each boundary is inclusive.
    assert band_membership(0.67, 1.0) == {"tight", "loose"}
    assert band_membership(1.5, 1.0) == {"tight", "loose"}
    assert band_membership(0.33, 1.0) == {"loose"}
    assert band_membership(3.0, 1.0) == {"loose"}
    # Just outside the loose band on both sides.
    assert band_membership(0.32, 1.0) == set()
    assert band_membership(3.01, 1.0) == set()
    # Tight items are always a subset of loose items (bands nest).
    assert band_membership(1.2, 1.0) == {"tight", "loose"}


def test_band_membership_floor_excludes_near_zero_baselines():
    # ratio would read as a tight match (1.0), but both values are below the floor.
    tiny = BAND_FLOOR / 2
    assert band_membership(tiny, tiny) == set()
    # One side below the floor, even if the other is not.
    assert band_membership(BAND_FLOOR / 2, 1.0) == set()


def test_band_membership_zero_third_does_not_raise():
    assert band_membership(0.5, 0.0) == set()


def test_items_in_band_filters_correctly():
    baselines = {
        "tight_item": (1.0, 1.0),          # ratio 1.0: tight and loose
        "loose_only_item": (2.0, 1.0),     # ratio 2.0: loose only
        "outside_item": (5.0, 1.0),        # ratio 5.0: neither
        "below_floor_item": (0.001, 0.001),  # ratio 1.0 but below floor: neither
    }
    assert items_in_band(baselines, "tight") == {"tight_item"}
    assert items_in_band(baselines, "loose") == {"tight_item", "loose_only_item"}


def test_bands_are_consistent_with_documented_constants():
    # Regression guard: if these drift, every printed band label in the report is wrong.
    assert TIGHT_BAND == (0.67, 1.5)
    assert LOOSE_BAND == (0.33, 3.0)
    assert BAND_FLOOR == 0.01
