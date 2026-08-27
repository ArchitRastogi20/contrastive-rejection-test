"""Tests for pilot.analyze_relation_stratum: hand-written fixtures only, no GPU or network."""

from __future__ import annotations

import json
import math

from pilot.analyze_relation_stratum import (
    PARENTAGE_ATTRIBUTES,
    continuous_contrast,
    continuous_pairs,
    discrete_contrast,
    is_parentage,
    item_attribute,
    load_part,
    split_by_stratum,
)


# ------------------------------------------------------------------------- stratum assignment


def test_parentage_attributes_are_exactly_child_father_mother():
    assert PARENTAGE_ATTRIBUTES == {"child", "father", "mother"}


def test_is_parentage_true_for_each_of_child_father_mother():
    assert is_parentage("child")
    assert is_parentage("father")
    assert is_parentage("mother")


def test_is_parentage_false_for_a_non_parentage_attribute():
    assert not is_parentage("date_of_birth")


def test_is_parentage_false_for_none():
    assert not is_parentage(None)


def _row(condition, attribute, *, chosen_is_edited=None, delta_p_edited=None, complete=True):
    return {
        "condition": condition, "attribute": attribute,
        "chosen_is_edited": chosen_is_edited, "delta_p_edited": delta_p_edited,
        "letter_probe": {"complete": complete},
    }


def test_item_attribute_reads_the_field_off_whichever_condition_is_present():
    conds = {"R1": _row("R1", "father"), "R2": _row("R2", "father")}
    assert item_attribute(conds) == "father"


def test_split_by_stratum_separates_parentage_from_non_parentage_and_pools_both():
    items = {
        "m::i1": {"R1": _row("R1", "child"), "R2": _row("R2", "child")},
        "m::i2": {"R1": _row("R1", "mother"), "R2": _row("R2", "mother")},
        "m::i3": {"R1": _row("R1", "father"), "R2": _row("R2", "father")},
        "m::i4": {"R1": _row("R1", "date_of_birth"), "R2": _row("R2", "date_of_birth")},
        "m::i5": {"R1": _row("R1", "director"), "R2": _row("R2", "director")},
    }
    strata = split_by_stratum(items)
    assert set(strata["parentage"]) == {"m::i1", "m::i2", "m::i3"}
    assert set(strata["non_parentage"]) == {"m::i4", "m::i5"}
    assert set(strata["pooled"]) == set(items)


# --------------------------------------------------------------------------- McNemar b/c, exact p


def test_discrete_contrast_b_c_counted_by_hand():
    """4 items: 2 favour a-only (b), 1 favours b-only (c), 1 concordant (ignored)."""
    items = {
        "m::i1": {"R1": _row("R1", "director", chosen_is_edited=True),
                  "R2": _row("R2", "director", chosen_is_edited=False)},
        "m::i2": {"R1": _row("R1", "director", chosen_is_edited=True),
                  "R2": _row("R2", "director", chosen_is_edited=False)},
        "m::i3": {"R1": _row("R1", "director", chosen_is_edited=False),
                  "R2": _row("R2", "director", chosen_is_edited=True)},
        "m::i4": {"R1": _row("R1", "director", chosen_is_edited=True),
                  "R2": _row("R2", "director", chosen_is_edited=True)},
    }
    d = discrete_contrast(items, "R1", "R2")
    assert d["b_a_only"] == 2
    assert d["c_b_only"] == 1
    assert d["n_discordant"] == 3


def test_discrete_contrast_exact_p_on_a_tiny_known_case():
    """b=1, c=0, n=1: exact two-sided binomial p = min(1, 2 * C(1,0) * 0.5**1) = 1.0."""
    items = {
        "m::i1": {"R1": _row("R1", "director", chosen_is_edited=True),
                  "R2": _row("R2", "director", chosen_is_edited=False)},
    }
    d = discrete_contrast(items, "R1", "R2")
    assert d["b_a_only"] == 1
    assert d["c_b_only"] == 0
    assert math.isclose(d["p_exact_two_sided"], 1.0, rel_tol=1e-12)


def test_discrete_contrast_exact_p_on_a_larger_known_case():
    """b=4, c=0, n=4: exact two-sided p = min(1, 2 * C(4,0) * 0.5**4) = min(1, 2/16) = 0.125."""
    items = {}
    for i in range(4):
        items[f"m::i{i}"] = {"R1": _row("R1", "director", chosen_is_edited=True),
                              "R2": _row("R2", "director", chosen_is_edited=False)}
    d = discrete_contrast(items, "R1", "R2")
    assert d["b_a_only"] == 4
    assert d["c_b_only"] == 0
    assert math.isclose(d["p_exact_two_sided"], 0.125, rel_tol=1e-9)


def test_discrete_contrast_n_zero_reports_p_one_and_no_discordant_pairs():
    items = {
        "m::i1": {"R1": _row("R1", "director", chosen_is_edited=True),
                  "R2": _row("R2", "director", chosen_is_edited=True)},
    }
    d = discrete_contrast(items, "R1", "R2")
    assert d["n_discordant"] == 0
    assert math.isclose(d["p_exact_two_sided"], 1.0, rel_tol=1e-12)


# --------------------------------------------------------------------- Haldane-Anscombe branch


def test_discrete_contrast_flags_haldane_anscombe_when_a_cell_is_zero():
    items = {}
    for i in range(5):
        items[f"m::i{i}"] = {"R1": _row("R1", "director", chosen_is_edited=True),
                              "R2": _row("R2", "director", chosen_is_edited=False)}
    d = discrete_contrast(items, "R1", "R2")
    assert d["c_b_only"] == 0
    assert d["or"]["haldane_anscombe"] is True
    # (5+0.5)/(0+0.5) = 11.0
    assert math.isclose(d["or"]["or"], 11.0, rel_tol=1e-9)


def test_discrete_contrast_does_not_correct_when_both_cells_are_nonzero():
    items = {
        "m::i1": {"R1": _row("R1", "director", chosen_is_edited=True),
                  "R2": _row("R2", "director", chosen_is_edited=False)},
        "m::i2": {"R1": _row("R1", "director", chosen_is_edited=True),
                  "R2": _row("R2", "director", chosen_is_edited=False)},
        "m::i3": {"R1": _row("R1", "director", chosen_is_edited=False),
                  "R2": _row("R2", "director", chosen_is_edited=True)},
    }
    d = discrete_contrast(items, "R1", "R2")
    assert d["or"]["haldane_anscombe"] is False
    assert math.isclose(d["or"]["or"], 2.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------- continuous measure


def test_continuous_pairs_requires_both_deltas_non_null_and_both_probes_complete():
    items = {
        # both sides usable
        "m::i1": {"R1": _row("R1", "director", delta_p_edited=0.3),
                  "R2": _row("R2", "director", delta_p_edited=0.1)},
        # R2 delta_p_edited is null -- excluded
        "m::i2": {"R1": _row("R1", "director", delta_p_edited=0.2),
                  "R2": _row("R2", "director", delta_p_edited=None)},
        # R1 probe incomplete -- excluded even though both deltas are present
        "m::i3": {"R1": _row("R1", "director", delta_p_edited=0.5, complete=False),
                  "R2": _row("R2", "director", delta_p_edited=0.1)},
    }
    diffs = continuous_pairs(items, "R1", "R2")
    assert diffs == [0.3 - 0.1]


def test_continuous_contrast_mean_matches_a_hand_computed_value():
    items = {
        "m::i1": {"R1": _row("R1", "director", delta_p_edited=0.4),
                  "R2": _row("R2", "director", delta_p_edited=0.1)},
        "m::i2": {"R1": _row("R1", "director", delta_p_edited=-0.2),
                  "R2": _row("R2", "director", delta_p_edited=0.0)},
    }
    c = continuous_contrast(items, "R1", "R2", seed=1)
    expected = ((0.4 - 0.1) + (-0.2 - 0.0)) / 2
    assert math.isclose(c["mean"], expected, rel_tol=1e-12)
    assert c["n"] == 2
    assert c["contrast"] == "R1-R2"


def test_continuous_contrast_on_no_pairs_is_all_none():
    items = {"m::i1": {"R1": _row("R1", "director", delta_p_edited=None)}}
    c = continuous_contrast(items, "R1", "R2", seed=1)
    assert c["mean"] is None
    assert c["n"] == 0


# --------------------------------------------------------------------------------- load_part


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_part_keys_by_model_and_item_id_and_reads_the_condition(tmp_path):
    part_dir = tmp_path / "exp3z"
    part_dir.mkdir()
    rows = [
        {"model": "m1", "item_id": "i1", "condition": "R0", "attribute": "father"},
        {"model": "m1", "item_id": "i1", "condition": "R1", "attribute": "father"},
        {"model": "m2", "item_id": "i1", "condition": "R0", "attribute": "director"},
    ]
    _write_jsonl(part_dir / "stage3_m1.jsonl", [r for r in rows if r["model"] == "m1"])
    _write_jsonl(part_dir / "stage3_m2.jsonl", [r for r in rows if r["model"] == "m2"])

    loaded = load_part(tmp_path, "exp3z")
    assert set(loaded) == {"m1::i1", "m2::i1"}
    assert loaded["m1::i1"]["R0"]["attribute"] == "father"
    assert loaded["m1::i1"]["R1"]["attribute"] == "father"
    # same item_id under a different model must not collide with m1's record.
    assert loaded["m2::i1"]["R0"]["attribute"] == "director"


# --------------------------------------------------------------- no stray control characters


def test_touched_files_contain_no_stray_control_characters():
    """A patch once wrote a literal backspace where a regex needed a word boundary -- see
    test_repair.py / test_extract.py / test_analyze_run3.py's identical guard, scoped here to
    the files this task touched."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    paths = [here.parent / "pilot" / "analyze_relation_stratum.py", here / "test_analyze_relation_stratum.py"]

    offenders = []
    for path in paths:
        for i, byte in enumerate(path.read_bytes()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((str(path), i, byte))
    assert offenders == []
