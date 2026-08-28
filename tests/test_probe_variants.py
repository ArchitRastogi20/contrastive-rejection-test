"""Tests for the probe-variant experiment (E2): the variant table, the completion/agreement
statistics, the budget check, and the end-to-end dry run.

Hand-written fixtures only; no GPU or network.
"""

from __future__ import annotations

import json

import pytest

from harness.models import append_letter_probe
from harness.probe_variants import (
    PREFILL_PAREN_STEM,
    PREFILL_STEM,
    VARIANTS,
    budget_check,
    completion_by,
    completion_by_two,
    estimate_gpu_seconds,
    main,
    top_letter,
    variant_agreement,
)

_CHAT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "Question: who?\n\nCandidates:\n\nA) X\nB) Y"},
]


# ------------------------------------------------------------------------------ the variant table


def test_every_variant_produces_a_well_formed_chat():
    for name, variant in VARIANTS.items():
        built = variant.build(_CHAT)
        assert isinstance(built, list) and len(built) >= 2, name
        for turn in built:
            assert set(turn) >= {"role", "content"}, name
            assert turn["role"] in ("system", "user", "assistant"), name
        # every turn up to and including the (possibly instruction-merged) user turn must still
        # start the same way the input chat did
        assert built[0] == _CHAT[0], name


def test_baseline_variant_reproduces_append_letter_probe_exactly():
    assert VARIANTS["baseline"].build(_CHAT) == append_letter_probe(_CHAT)


def test_prefill_variants_end_in_an_assistant_turn_with_the_declared_stem():
    stem_chat = VARIANTS["prefill_stem"].build(_CHAT)
    assert stem_chat[-1] == {"role": "assistant", "content": PREFILL_STEM}

    paren_chat = VARIANTS["prefill_paren"].build(_CHAT)
    assert paren_chat[-1] == {"role": "assistant", "content": PREFILL_PAREN_STEM}


def test_prefill_variants_still_carry_the_letter_probe_instruction_on_the_user_turn():
    """The prefilled assistant turn is additional, not a replacement for the existing
    instruction -- both variants must still fold the "single letter" instruction into the user
    turn exactly as the baseline does, so the only thing that differs between arms is the
    continuation, never the instruction the model was given."""
    for name in ("prefill_stem", "prefill_paren"):
        built = VARIANTS[name].build(_CHAT)
        user_turns = [t for t in built if t["role"] == "user"]
        assert len(user_turns) == 1
        assert "single letter" in user_turns[0]["content"]


def test_variant_table_has_at_least_baseline_plus_two_others():
    assert "baseline" in VARIANTS
    assert len(VARIANTS) >= 3


# ------------------------------------------------------------------------------------ top_letter


def test_top_letter_is_the_argmax():
    from harness.models import LetterProbRead

    read = LetterProbRead(
        candidates=["A", "B", "C"], raw_logprobs={"A": -1.0, "B": -0.1, "C": -2.0},
        probs={"A": 0.2, "B": 0.7, "C": 0.1}, complete=True, backend="test",
    )
    assert top_letter(read) == "B"


def test_top_letter_is_none_when_nothing_was_read():
    from harness.models import LetterProbRead

    read = LetterProbRead(
        candidates=["A", "B"], raw_logprobs={"A": None, "B": None}, probs={},
        complete=False, backend="test",
    )
    assert top_letter(read) is None


# --------------------------------------------------------------------------------- statistics


def _row(model, item_id, condition, variant, complete, top):
    return {"model": model, "item_id": item_id, "condition": condition, "variant": variant,
            "complete": complete, "top_letter": top}


def test_completion_by_variant():
    rows = [
        _row("m", "i1", "R0", "baseline", True, "A"),
        _row("m", "i1", "R0", "prefill_stem", True, "A"),
        _row("m", "i2", "R0", "baseline", False, None),
        _row("m", "i2", "R0", "prefill_stem", True, "B"),
    ]
    out = completion_by(rows, "variant")
    assert out["baseline"] == {"n": 2, "complete": 1, "rate": 0.5}
    assert out["prefill_stem"] == {"n": 2, "complete": 2, "rate": 1.0}


def test_completion_by_two_nests_correctly():
    rows = [
        _row("m", "i1", "R0", "baseline", True, "A"),
        _row("m", "i1", "R1", "baseline", False, None),
    ]
    out = completion_by_two(rows, "variant", "condition")
    assert out["baseline"]["R0"] == {"n": 1, "complete": 1, "rate": 1.0}
    assert out["baseline"]["R1"] == {"n": 1, "complete": 0, "rate": 0.0}


def test_variant_agreement_excludes_pairs_incomplete_on_either_side():
    rows = [
        _row("m", "i1", "R0", "baseline", True, "A"),
        _row("m", "i1", "R0", "prefill_stem", True, "A"),  # agree
        _row("m", "i2", "R0", "baseline", True, "B"),
        _row("m", "i2", "R0", "prefill_stem", True, "C"),  # disagree
        _row("m", "i3", "R0", "baseline", False, None),
        _row("m", "i3", "R0", "prefill_stem", True, "A"),  # excluded: baseline incomplete
    ]
    out = variant_agreement(rows)
    key = "baseline_vs_prefill_stem"
    assert out[key]["n_both_complete"] == 2
    assert out[key]["n_agree"] == 1
    assert out[key]["rate"] == 0.5


def test_variant_agreement_covers_every_pair():
    rows = [_row("m", "i1", "R0", name, True, "A") for name in VARIANTS]
    out = variant_agreement(rows)
    names = list(VARIANTS)
    expected_pairs = {
        f"{names[i]}_vs_{names[j]}" for i in range(len(names)) for j in range(i + 1, len(names))
    }
    assert set(out) == expected_pairs


# --------------------------------------------------------------------------------- budget check


def test_estimate_scales_with_rows_and_variant_count():
    assert estimate_gpu_seconds(10) == 10 * estimate_gpu_seconds(1)


def test_budget_check_refuses_when_estimate_exceeds_remaining():
    out = budget_check(1_000_000, spent=0.0, allowance=10.0)
    assert out["ok"] is False


def test_budget_check_passes_within_a_generous_allowance():
    out = budget_check(1, spent=0.0, allowance=1_000_000.0)
    assert out["ok"] is True


# ------------------------------------------------------------------------------------ dry run


def test_dry_run_end_to_end_writes_rows_and_a_summary(tmp_path):
    rc = main(["--dry-run", "--out-dir", str(tmp_path)])
    assert rc == 0

    jsonl_files = list(tmp_path.glob("probe_variants_*.jsonl"))
    assert len(jsonl_files) == 1
    rows = [json.loads(l) for l in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
    assert rows

    seen_variants = {row["variant"] for row in rows}
    assert seen_variants == set(VARIANTS)
    for row in rows:
        assert row["condition"] in ("R0", "R1", "R2", "R3", "R4")
        assert "complete" in row and "probs" in row and "raw_logprobs" in row

    summary_path = tmp_path / "probe_variants_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["models"] == ["stub"]
    assert set(summary["variants"]) == set(VARIANTS)


def test_dry_run_respects_limit(tmp_path):
    rc = main(["--dry-run", "--limit", "0", "--out-dir", str(tmp_path)])
    assert rc == 0


# --------------------------------------------------------------------------------- cleanliness


def test_touched_files_contain_no_stray_control_characters():
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    paths = [here.parent / "harness" / "probe_variants.py", here / "test_probe_variants.py"]

    offenders = []
    for path in paths:
        for i, byte in enumerate(path.read_bytes()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((str(path), i, byte))
    assert offenders == []
