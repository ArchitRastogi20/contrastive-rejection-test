"""Unit tests for the offline condition-specific probe-missingness audit."""

from harness.audit_probe_missingness import (
    CONDITIONS,
    condition_counts,
    paired_complete_counts,
    permutation_p_value,
    rate_spread,
)


def test_condition_counts_and_spread_are_hand_checkable():
    items = {
        "i1": {"R0": True, "R1": True, "R2": False, "R3": False, "R4": True},
        "i2": {"R0": True, "R1": False, "R2": False, "R3": False, "R4": True},
    }
    counts = condition_counts(items)
    assert counts["R0"] == {"complete": 2, "incomplete": 0}
    assert counts["R2"] == {"complete": 0, "incomplete": 2}
    assert rate_spread(counts) == 1.0


def test_paired_complete_counts_requires_both_members_of_a_contrast():
    items = {
        "i1": {"R0": True, "R1": True, "R2": True, "R3": False, "R4": True},
        "i2": {"R0": True, "R1": True, "R2": False, "R3": True, "R4": True},
    }
    assert paired_complete_counts(items) == {"R1-R2": 1, "R3-R4": 1, "R1-R3": 1, "R2-R4": 1}


def test_permutation_is_deterministic_and_preserves_item_level_completion_counts():
    items = {
        f"i{i}": dict(zip(CONDITIONS, [True, True, False, False, True]))
        for i in range(8)
    }
    first = permutation_p_value(items, seed=123, resamples=1000)
    second = permutation_p_value(items, seed=123, resamples=1000)
    assert first == second
    assert first["n_items"] == 8
    assert 0.0 <= first["p_value"] <= 1.0
