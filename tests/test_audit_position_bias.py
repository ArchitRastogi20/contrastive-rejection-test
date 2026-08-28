"""Unit tests for the offline option-letter-effects audit."""

from harness.audit_position_bias import (
    RESAMPLES,
    SEED,
    chi_square_stat,
    count_excluded_r0_choices,
    letter_mismatch_count,
    letter_of_pair,
    margin_outcomes_logodds,
    part_letters,
    stratify_by_letter,
    tabulate_r0_choices,
    uniform_letter_test,
)


def _probe(candidates, logprobs, complete=True):
    return {
        "candidates": candidates,
        "raw_logprobs": {c: logprobs.get(c) for c in candidates},
        "probs": {},
        "complete": complete,
    }


# ------------------------------------------------------------------------------- section 1


def test_part_letters_reads_candidates_even_when_probe_incomplete():
    models = {
        "m1": {
            "i1": {"R0": {"choice": "A", "letter_probe": _probe(["A", "B", "C"], {}, complete=False)}},
        }
    }
    assert part_letters(models) == ["A", "B", "C"]


def test_tabulate_r0_choices_counts_only_known_letters():
    items = {
        "i1": {"R0": {"choice": "A"}},
        "i2": {"R0": {"choice": "A"}},
        "i3": {"R0": {"choice": "B"}},
        "i4": {"R0": {"choice": None}},
        "i5": {"R0": {"choice": "Z"}},
        "i6": {},  # no R0 at all
    }
    letters = ["A", "B", "C"]
    assert tabulate_r0_choices(items, letters) == {"A": 2, "B": 1, "C": 0}
    assert count_excluded_r0_choices(items, letters) == 2  # None and "Z"


def test_chi_square_stat_is_zero_for_a_perfectly_uniform_sample():
    counts = {"A": 10, "B": 10, "C": 10, "D": 10}
    assert chi_square_stat(counts, ["A", "B", "C", "D"]) == 0.0


def test_chi_square_stat_matches_hand_computation_for_a_skewed_sample():
    # expected = 40/4 = 10 each; chi2 = sum((o-e)^2/e)
    counts = {"A": 40, "B": 0, "C": 0, "D": 0}
    expected_chi2 = ((40 - 10) ** 2) / 10 + 3 * ((0 - 10) ** 2 / 10)
    assert chi_square_stat(counts, ["A", "B", "C", "D"]) == expected_chi2


def test_uniform_letter_test_is_deterministic_and_flags_a_maximally_skewed_sample():
    letters = ["A", "B", "C", "D"]
    skewed = {"A": 40, "B": 0, "C": 0, "D": 0}
    first = uniform_letter_test(skewed, letters, seed=7, resamples=500)
    second = uniform_letter_test(skewed, letters, seed=7, resamples=500)
    assert first == second
    assert first["n"] == 40
    # No random draw of 40 uniform letters into 4 bins can be more extreme than "all in one bin",
    # so this must hit the smallest possible p-value the resample count allows.
    assert first["p_value"] == 1 / 501

    uniform = {"A": 10, "B": 10, "C": 10, "D": 10}
    null_case = uniform_letter_test(uniform, letters, seed=7, resamples=500)
    assert null_case["chi2"] == 0.0
    # A perfectly uniform observation is at least as extreme as almost every resample, including
    # itself, so its p-value should sit at 1.0.
    assert null_case["p_value"] == 1.0


def test_uniform_letter_test_handles_zero_total_without_dividing_by_zero():
    result = uniform_letter_test({"A": 0, "B": 0}, ["A", "B"])
    assert result == {"chi2": None, "p_value": None, "n": 0, "resamples": RESAMPLES, "seed": SEED}


# ------------------------------------------------------------------------------- section 2


def test_letter_of_pair_requires_agreement_between_both_sides():
    conds_ok = {"R1": {"edited_letter": "B"}, "R2": {"edited_letter": "B"}}
    conds_missing = {"R1": {"edited_letter": "B"}}
    conds_mismatch = {"R1": {"edited_letter": "B"}, "R2": {"edited_letter": "C"}}
    assert letter_of_pair(conds_ok, "R1", "R2") == "B"
    assert letter_of_pair(conds_missing, "R1", "R2") is None
    assert letter_of_pair(conds_mismatch, "R1", "R2") is None


def test_letter_mismatch_count_counts_only_true_disagreements():
    pooled = {
        "i1": {"R1": {"edited_letter": "A"}, "R2": {"edited_letter": "A"}},
        "i2": {"R1": {"edited_letter": "A"}, "R2": {"edited_letter": "B"}},
        "i3": {"R1": {"edited_letter": "A"}},  # one-sided, not a mismatch
    }
    assert letter_mismatch_count(pooled, "R1", "R2") == 1


def test_stratify_by_letter_splits_items_into_the_right_buckets():
    pooled = {
        "i1": {
            "R1": {"edited_letter": "A", "chosen_is_edited": True},
            "R2": {"edited_letter": "A", "chosen_is_edited": False},
        },
        "i2": {
            "R1": {"edited_letter": "B", "chosen_is_edited": False},
            "R2": {"edited_letter": "B", "chosen_is_edited": False},
        },
        "i3": {
            "R1": {"edited_letter": "A", "chosen_is_edited": False},
            "R2": {"edited_letter": "A", "chosen_is_edited": False},
        },
    }
    strata = stratify_by_letter(pooled, "R1", "R2")
    assert set(strata.keys()) == {"A", "B"}
    assert set(strata["A"].keys()) == {"i1", "i3"}
    assert set(strata["B"].keys()) == {"i2"}
    assert strata["A"]["i1"] == {"R1": True, "R2": False}


# ------------------------------------------------------------------------------- section 3


def test_margin_outcomes_logodds_matches_hand_computation():
    items = {
        "i1": {
            "R0": {
                "letter_probe": _probe(["A", "B"], {"A": -1.0, "B": -2.0}, complete=True),
            },
            "R1": {
                "edited_letter": "A",
                "letter_probe": _probe(["A", "B"], {"A": -0.5, "B": -3.0}, complete=True),
            },
        }
    }
    out = margin_outcomes_logodds(items)
    # R0 margin for A: -1.0 - (-2.0) = 1.0. R1 margin for A: -0.5 - (-3.0) = 2.5.
    # delta = 2.5 - 1.0 = 1.5
    assert out["i1"]["R1"]["delta_margin_logodds"] == 1.5


def test_margin_outcomes_logodds_excludes_a_probe_that_never_completes():
    # Mirrors the real DeepSeek-R1-Distill row: raw_logprobs is all null, complete is False.
    items = {
        "i1": {
            "R0": {
                "letter_probe": _probe(["A", "B"], {"A": None, "B": None}, complete=False),
            },
            "R1": {
                "edited_letter": "A",
                "letter_probe": _probe(["A", "B"], {"A": None, "B": None}, complete=False),
            },
        },
        "i2": {
            "R0": {
                "letter_probe": _probe(["A", "B"], {"A": -1.0, "B": -2.0}, complete=True),
            },
            "R1": {
                "edited_letter": "A",
                "letter_probe": _probe(["A", "B"], {"A": None, "B": None}, complete=False),
            },
        },
    }
    out = margin_outcomes_logodds(items)
    assert out == {}


def test_margin_outcomes_logodds_requires_r0_complete_but_lets_conditions_vary():
    items = {
        "i1": {
            "R0": {"letter_probe": _probe(["A", "B"], {"A": -1.0, "B": -2.0}, complete=True)},
            "R1": {"edited_letter": "A",
                   "letter_probe": _probe(["A", "B"], {"A": -0.2, "B": -1.0}, complete=True)},
            "R2": {"edited_letter": "A",
                   "letter_probe": _probe(["A", "B"], {"A": None, "B": None}, complete=False)},
        }
    }
    out = margin_outcomes_logodds(items)
    assert set(out["i1"].keys()) == {"R1"}
