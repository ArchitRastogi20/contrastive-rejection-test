"""Tests for pilot.r3_recency: derives the R3 recency measure and target letter.

Hand-written fixtures only; no GPU or network."""

import json

from pilot.r3_recency import (
    CONDITIONS,
    LetterUnavailable,
    analyse,
    derive_r3_letter,
    main,
    resolve_targets,
)


# --------------------------------------------------------------- R3-edited-letter derivation


def test_derive_r3_letter_matches_repair_py_rule():
    # This is run 1's own item 1a114c5e...: rival A, stage-1 choice B, gold B, 4 options.
    # excluded = {A, B}; the first free letter in order is C -- and the saved stage-3 R3
    # response for this exact item does read "The correct candidate is C) The Majesty of the
    # Law", so this is not just an internally-consistent rule, it is the rule repair.py used.
    assert derive_r3_letter("A", "B", "B", 4) == "C"


def test_derive_r3_letter_when_choice_and_gold_coincide():
    # A model that answered correctly still excludes only two distinct letters, so the first
    # free letter shifts left: excluded = {A, C} (rival A, choice C, gold C) -> B is free.
    assert derive_r3_letter("A", "C", "C", 4) == "B"


def test_derive_r3_letter_none_when_every_letter_excluded():
    # Only 2 options and both letters are excluded -- no R3 target exists. Must not guess.
    assert derive_r3_letter("A", "A", "B", 2) is None


# ---------------------------------------------------------------------- unavailable, not guessed


def test_resolve_targets_reports_missing_gold_letter_rather_than_guessing():
    stage1_row = {"choice": "B", "option_titles": ["a", "b", "c", "d"]}  # no gold_letter
    try:
        resolve_targets(stage1_row, rival_letter="A")
    except LetterUnavailable as exc:
        assert "gold_letter" in str(exc)
    else:
        raise AssertionError("expected LetterUnavailable when gold_letter is absent")


def test_resolve_targets_reports_inconsistent_rival_letter():
    stage1_row = {"choice": "B", "gold_letter": "B", "option_titles": ["a", "b", "c", "d"]}
    try:
        resolve_targets(stage1_row, rival_letter=None)  # analyse() passes None on disagreement
    except LetterUnavailable as exc:
        assert "rival_letter" in str(exc)
    else:
        raise AssertionError("expected LetterUnavailable when rival_letter is unusable")


# -------------------------------------------------------------------------------- fixture I/O


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _stage1_row(model, item_id, *, choice, gold_letter, option_titles):
    return {"model": model, "item_id": item_id, "choice": choice, "gold_letter": gold_letter,
            "option_titles": option_titles}


def _stage2_row(model, item_id, *, built):
    return {"model": model, "item_id": item_id, "built": built}


def _stage3_row(model, item_id, condition, *, rival_letter, choice):
    return {"model": model, "item_id": item_id, "condition": condition,
            "rival_letter": rival_letter, "choice": choice,
            "chosen_is_repaired_rival": None if choice is None else choice == rival_letter}


def _build_fixture(exp_dir):
    """Two models, three items, hand-computable by the rules above:

    model M1 / item "flip": rival A, stage-1 choice B, gold B -> R3 letter C.
        R0 chooses B (untouched, matches stage-1 exactly -- the R0 integrity check).
        R1 chooses A (the rival, repaired -- the primary measure hits).
        R2 chooses B (control edit does not move it).
        R3 chooses C (the option R3 actually edited -- the recency measure hits).

    model M1 / item "no_recency": rival A, stage-1 choice B, gold B -> R3 letter C, same as
    above, but every condition reproduces B: no movement anywhere, including R3.

    model M2 / item "unreadable": rival A, stage-1 choice C, gold D -> excluded {A, C, D},
    R3 letter B. R2's response could not be parsed (choice None), covering the "unreadable"
    branch: it must count separately, not as a miss.
    """
    opts = ["t0", "t1", "t2", "t3"]
    s1 = [
        _stage1_row("M1", "flip", choice="B", gold_letter="B", option_titles=opts),
        _stage1_row("M1", "no_recency", choice="B", gold_letter="B", option_titles=opts),
        _stage1_row("M2", "unreadable", choice="C", gold_letter="D", option_titles=opts),
    ]
    s2 = [
        _stage2_row("M1", "flip", built=True),
        _stage2_row("M1", "no_recency", built=True),
        _stage2_row("M2", "unreadable", built=True),
        _stage2_row("M1", "not_built", built=False),  # must be ignored entirely
    ]
    s3 = []
    s3 += [
        _stage3_row("M1", "flip", "R0", rival_letter="A", choice="B"),
        _stage3_row("M1", "flip", "R1", rival_letter="A", choice="A"),
        _stage3_row("M1", "flip", "R2", rival_letter="A", choice="B"),
        _stage3_row("M1", "flip", "R3", rival_letter="A", choice="C"),
    ]
    s3 += [
        _stage3_row("M1", "no_recency", "R0", rival_letter="A", choice="B"),
        _stage3_row("M1", "no_recency", "R1", rival_letter="A", choice="B"),
        _stage3_row("M1", "no_recency", "R2", rival_letter="A", choice="B"),
        _stage3_row("M1", "no_recency", "R3", rival_letter="A", choice="B"),
    ]
    s3 += [
        _stage3_row("M2", "unreadable", "R0", rival_letter="A", choice="C"),
        _stage3_row("M2", "unreadable", "R1", rival_letter="A", choice="A"),
        _stage3_row("M2", "unreadable", "R2", rival_letter="A", choice=None),
        _stage3_row("M2", "unreadable", "R3", rival_letter="A", choice="B"),
    ]

    _write_jsonl(exp_dir / "stage1_m1.jsonl", [r for r in s1 if r["model"] == "M1"])
    _write_jsonl(exp_dir / "stage1_m2.jsonl", [r for r in s1 if r["model"] == "M2"])
    _write_jsonl(exp_dir / "stage2_m1.jsonl", [r for r in s2 if r["model"] == "M1"])
    _write_jsonl(exp_dir / "stage2_m2.jsonl", [r for r in s2 if r["model"] == "M2"])
    _write_jsonl(exp_dir / "stage3_m1.jsonl", [r for r in s3 if r["model"] == "M1"])
    _write_jsonl(exp_dir / "stage3_m2.jsonl", [r for r in s3 if r["model"] == "M2"])


# ------------------------------------------------------------------------------- counting logic


def test_analyse_counts_match_hand_computation(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    _build_fixture(exp_dir)
    summary = analyse(exp_dir)

    assert summary["items_considered"] == 3  # "not_built" must not be counted
    assert summary["items_derived"] == 3
    assert summary["items_unavailable"] == []

    m1 = summary["per_model"]["M1"]
    # "flip" and "no_recency" both have rival A, choice B -> R0/R3 target C, R1/R2 target A.
    assert m1["R0"] == {"target_role": "r3_edited_option", "n": 2, "chose_edited_option": 0,
                         "chose_other": 2, "unreadable": 0, "rate": 0.0}
    assert m1["R1"]["chose_edited_option"] == 1  # "flip" chose A, "no_recency" chose B
    assert m1["R1"]["n"] == 2
    assert m1["R2"]["chose_edited_option"] == 0  # neither chose A under R2
    assert m1["R3"]["chose_edited_option"] == 1  # only "flip" chose C
    assert m1["R3"]["rate"] == 0.5

    m2 = summary["per_model"]["M2"]
    assert m2["R2"] == {"target_role": "rival_option", "n": 1, "chose_edited_option": 0,
                         "chose_other": 0, "unreadable": 1, "rate": 0.0}
    assert m2["R3"]["chose_edited_option"] == 1  # target B, chose B

    pooled = summary["pooled"]
    assert pooled["R3"]["n"] == 3
    assert pooled["R3"]["chose_edited_option"] == 2  # "flip" and "unreadable", not "no_recency"


def test_analyse_reports_incomplete_stage3_as_unavailable_not_guessed(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    _stage1s = [_stage1_row("M1", "partial", choice="B", gold_letter="B",
                             option_titles=["t0", "t1", "t2", "t3"])]
    _write_jsonl(exp_dir / "stage1_m1.jsonl", _stage1s)
    _write_jsonl(exp_dir / "stage2_m1.jsonl", [_stage2_row("M1", "partial", built=True)])
    # Only R0-R2 present; R3's stage-3 record is missing entirely.
    partial = [
        _stage3_row("M1", "partial", "R0", rival_letter="A", choice="B"),
        _stage3_row("M1", "partial", "R1", rival_letter="A", choice="A"),
        _stage3_row("M1", "partial", "R2", rival_letter="A", choice="B"),
    ]
    _write_jsonl(exp_dir / "stage3_m1.jsonl", partial)

    summary = analyse(exp_dir)
    assert summary["items_considered"] == 1
    assert summary["items_derived"] == 0
    assert len(summary["items_unavailable"]) == 1
    reason = summary["items_unavailable"][0]["reason"]
    assert "R0-R3" in reason
    # No condition gained a phantom count for this item.
    assert summary["pooled"]["R0"]["n"] == 0


# ------------------------------------------------------------- recorded letter, preferred over derived


def _stage3_row_with_letter(model, item_id, condition, *, rival_letter, choice, edited_letter):
    """Like `_stage3_row`, plus the two fields run 2's `run_stage3` adds. Run 1's own rows never
    carry these -- `_stage3_row` alone (used everywhere above) is what a run-1-style record
    looks like, and every test above it already exercises the derivation fallback on exactly
    that shape."""
    row = _stage3_row(model, item_id, condition, rival_letter=rival_letter, choice=choice)
    row["edited_letter"] = edited_letter
    row["chosen_is_edited"] = (
        None if choice is None or edited_letter is None else choice == edited_letter
    )
    return row


def test_run1_style_records_without_edited_letter_still_derive():
    from pilot.r3_recency import _recorded_targets

    conds = {
        "R0": _stage3_row("M1", "x", "R0", rival_letter="A", choice="B"),
        "R1": _stage3_row("M1", "x", "R1", rival_letter="A", choice="A"),
        "R2": _stage3_row("M1", "x", "R2", rival_letter="A", choice="B"),
        "R3": _stage3_row("M1", "x", "R3", rival_letter="A", choice="C"),
    }
    # None of these rows carry `edited_letter` at all -- run 1's exact shape.
    assert _recorded_targets(conds) is None


def test_recorded_edited_letter_is_preferred_when_present(tmp_path):
    """Derivation (excluded={A, B}, 4 options) would give R3's letter as "C". The recorded
    `edited_letter` says "D" instead -- deliberately different, so which one `analyse()` used
    is observable in the resulting counts rather than merely asserted of a helper in isolation.
    """
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    opts = ["t0", "t1", "t2", "t3"]
    s1 = [_stage1_row("M1", "prefer", choice="B", gold_letter="B", option_titles=opts)]
    s2 = [_stage2_row("M1", "prefer", built=True)]
    s3 = [
        _stage3_row_with_letter(
            "M1", "prefer", "R0", rival_letter="A", choice="D", edited_letter=None
        ),
        _stage3_row_with_letter(
            "M1", "prefer", "R1", rival_letter="A", choice="A", edited_letter="A"
        ),
        _stage3_row_with_letter(
            "M1", "prefer", "R2", rival_letter="A", choice="B", edited_letter="A"
        ),
        _stage3_row_with_letter(
            "M1", "prefer", "R3", rival_letter="A", choice="D", edited_letter="D"
        ),
    ]
    _write_jsonl(exp_dir / "stage1_m1.jsonl", s1)
    _write_jsonl(exp_dir / "stage2_m1.jsonl", s2)
    _write_jsonl(exp_dir / "stage3_m1.jsonl", s3)

    summary = analyse(exp_dir)
    m1 = summary["per_model"]["M1"]
    # Under derivation (letter "C"), both R0 and R3's choice of "D" would be a miss. Using the
    # recorded "D" instead, both hit -- proof the recorded letter, not the derived one, won.
    assert m1["R0"]["chose_edited_option"] == 1
    assert m1["R3"]["chose_edited_option"] == 1
    assert summary["items_unavailable"] == []
    assert summary["consistency_warnings"] == []


# ----------------------------------------------------------------------------------- determinism


def test_analyse_is_deterministic(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    _build_fixture(exp_dir)
    first = analyse(exp_dir)
    second = analyse(exp_dir)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_main_writes_deterministic_summary_body(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    _build_fixture(exp_dir)
    out1, out2 = tmp_path / "out1.json", tmp_path / "out2.json"
    assert main(["--exp-dir", str(exp_dir), "--out", str(out1)]) == 0
    assert main(["--exp-dir", str(exp_dir), "--out", str(out2)]) == 0

    body1 = json.loads(out1.read_text(encoding="utf-8"))
    body2 = json.loads(out2.read_text(encoding="utf-8"))
    body1.pop("generated_utc")
    body2.pop("generated_utc")
    assert body1 == body2


def test_main_refuses_to_write_inside_exp_dir(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    _build_fixture(exp_dir)
    inside = exp_dir / "r3_recency_summary.json"
    assert main(["--exp-dir", str(exp_dir), "--out", str(inside)]) != 0
    assert not inside.exists()


# -------------------------------------------------------------------------------------------
# `CONDITIONS` is imported above purely so a rename of the tuple breaks this test file loudly.
assert CONDITIONS == ("R0", "R1", "R2", "R3")


# --------------------------------------------------------------- no stray control characters


def test_new_files_contain_no_stray_control_characters():
    """A patch once wrote a literal backspace where the regex needed a word boundary.

    The pattern still compiled, matched nothing, and silently zeroed a whole measurement. Cheap
    to check for, invisible to read for -- see test_repair.py / test_extract.py's identical
    guard, scoped here to just the two files this task added.
    """
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    paths = [here.parent / "pilot" / "r3_recency.py", here / "test_r3_recency.py"]

    offenders = []
    for path in paths:
        for i, byte in enumerate(path.read_bytes()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((str(path), i, byte))
    assert offenders == []
