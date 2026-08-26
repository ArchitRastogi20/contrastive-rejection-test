"""Tests for the inserted-sentence surprisal experiment (E1): `pilot.models.find_inserted_span`
and the new `prompt_token_logprobs` backend surface, plus `pilot.surprisal`'s arithmetic, budget
check and end-to-end dry run.

Hand-written fixtures only; no GPU or network.
"""

from __future__ import annotations

import json
import math

import pytest

from pilot.models import PromptLogprobs, StubBackend, find_inserted_span
from pilot.surprisal import (
    budget_check,
    condition_summary,
    estimate_gpu_seconds,
    main,
    paired_contrast,
    span_surprisal,
)

# --------------------------------------------------------------------- find_inserted_span


def test_identical_sequences_have_an_empty_span_at_the_end():
    base = [1, 2, 3, 4]
    start, end = find_inserted_span(base, list(base))
    assert start == end == len(base)


def test_pure_suffix_insertion_is_found():
    base = [1, 2, 3]
    edited = [1, 2, 3, 4, 5]
    assert find_inserted_span(base, edited) == (3, 5)


def test_pure_prefix_insertion_is_found():
    base = [3, 4, 5]
    edited = [1, 2, 3, 4, 5]
    assert find_inserted_span(base, edited) == (0, 2)


def test_middle_insertion_with_an_unchanged_tail_is_found():
    """The insertion lands in the middle of the sequence (an earlier option's profile grew,
    but later options and the task instructions after it are unchanged) -- a pure
    common-prefix comparison would wrongly report everything from the insertion point to the
    end as new; the common suffix must be recognised too."""
    base = [10, 11, 12, 13, 14, 15]
    edited = [10, 11, 99, 98, 97, 12, 13, 14, 15]
    assert find_inserted_span(base, edited) == (2, 5)


def test_inserted_tokens_recurring_elsewhere_in_the_sequence_do_not_confuse_the_finder():
    """The correctness-critical case: the inserted span's own token values (5, 6) already occur
    earlier in the base sequence. A content-based search (find where "5, 6" occurs) would find
    the wrong, earlier occurrence; a positional prefix/suffix diff cannot, because it never
    looks at what the tokens *are*, only at where the two sequences first and last agree."""
    base = [1, 5, 6, 2, 3, 4]
    edited = [1, 5, 6, 2, 5, 6, 3, 4]
    start, end = find_inserted_span(base, edited)
    assert (start, end) == (4, 6)
    assert edited[start:end] == [5, 6]


def test_inserted_tokens_matching_the_whole_base_sequence_still_isolate_correctly():
    """A harder variant of the recurrence case: the base sequence itself is a palindrome-like
    repeat, so both the prefix and a naive suffix search could plausibly claim more than the
    true insertion. The prefix/suffix diff must still land exactly on the true inserted middle."""
    base = [7, 7, 7]
    edited = [7, 7, 9, 9, 7]
    start, end = find_inserted_span(base, edited)
    assert (start, end) == (2, 4)
    assert edited[start:end] == [9, 9]


def test_span_finder_works_on_string_tokens_not_just_ints():
    """`find_inserted_span` is used on real tokenizer output (ids) and on the stub's
    whitespace-split "tokens" (strings) alike -- it must not assume int arithmetic."""
    base = ["The", "cat", "sat", "."]
    edited = ["The", "cat", "sat", "on", "the", "mat", "."]
    assert find_inserted_span(base, edited) == (3, 6)


# ----------------------------------------------------------------------------- span_surprisal


def _read(token_ids, logprobs):
    tokens = [str(t) for t in token_ids]
    complete = all(lp is not None for lp in logprobs[1:])
    return PromptLogprobs(token_ids=token_ids, tokens=tokens, logprobs=logprobs,
                           backend="test", complete=complete, detail={})


def test_span_surprisal_matches_hand_computed_mean_and_total():
    # base: [1, 2, 3]; edited inserts two tokens (8, 9) after position 1.
    base = _read([1, 2, 3], [None, -0.1, -0.2])
    edited = _read([1, 2, 8, 9, 3], [None, -0.1, -1.0, -3.0, -0.2])

    result = span_surprisal(base, edited)

    assert (result.span_start, result.span_end) == (2, 4)
    assert result.n_tokens == 2
    assert result.complete is True
    # NLL is -logprob: token at position 2 has logprob -1.0 -> nll 1.0; position 3 has -3.0 -> 3.0
    assert result.per_token_nll == [1.0, 3.0]
    assert math.isclose(result.total_nll, 4.0, rel_tol=1e-12)
    assert math.isclose(result.mean_nll, 2.0, rel_tol=1e-12)
    assert result.inserted_tokens == ["8", "9"]


def test_span_surprisal_is_incomplete_when_a_span_logprob_is_missing():
    base = _read([1, 2, 3], [None, -0.1, -0.2])
    edited = _read([1, 2, 8, 9, 3], [None, -0.1, None, -3.0, -0.2])

    result = span_surprisal(base, edited)

    assert result.complete is False
    assert result.mean_nll is None
    assert result.total_nll is None
    assert result.per_token_nll is None
    assert result.n_tokens == 2  # the span itself is still reported, just not scoreable


def test_span_surprisal_is_incomplete_when_nothing_was_inserted():
    base = _read([1, 2, 3], [None, -0.1, -0.2])
    edited = _read([1, 2, 3], [None, -0.1, -0.2])

    result = span_surprisal(base, edited)

    assert result.n_tokens == 0
    assert result.complete is False
    assert result.mean_nll is None


# ----------------------------------------------------------------------------- condition_summary


def test_condition_summary_averages_only_complete_reads():
    rows = [
        {"condition": "R1", "complete": True, "mean_nll": 1.0},
        {"condition": "R1", "complete": True, "mean_nll": 3.0},
        {"condition": "R1", "complete": False, "mean_nll": None},
        {"condition": "R2", "complete": True, "mean_nll": 0.5},
    ]
    out = condition_summary(rows)

    assert out["R1"]["n"] == 2
    assert math.isclose(out["R1"]["mean_nll"], 2.0, rel_tol=1e-12)
    assert out["R2"]["n"] == 1
    assert out["R3"]["n"] == 0
    assert out["R3"]["mean_nll"] is None


def test_paired_contrast_excludes_items_incomplete_on_either_side():
    rows = [
        {"item_id": "i1", "condition": "R1", "complete": True, "mean_nll": 2.0},
        {"item_id": "i1", "condition": "R2", "complete": True, "mean_nll": 0.5},
        {"item_id": "i2", "condition": "R1", "complete": True, "mean_nll": 1.0},
        {"item_id": "i2", "condition": "R2", "complete": False, "mean_nll": None},
    ]
    result = paired_contrast(rows, "R1", "R2", seed=1)

    assert result["n"] == 1
    assert math.isclose(result["mean"], 2.0 - 0.5, rel_tol=1e-12)
    assert result["contrast"] == "R1-R2"


# --------------------------------------------------------------------------------- budget check


def test_estimate_scales_with_item_condition_reads():
    assert estimate_gpu_seconds(10) == 10 * estimate_gpu_seconds(1)


def test_budget_check_passes_when_estimate_fits_remaining():
    out = budget_check(10, spent=0.0, allowance=1000.0)
    assert out["ok"] is True
    assert out["remaining_s"] == 1000.0


def test_budget_check_refuses_when_estimate_exceeds_remaining():
    # A huge item count against a tiny remaining allowance must refuse.
    out = budget_check(1_000_000, spent=0.0, allowance=10.0)
    assert out["ok"] is False
    assert out["estimate_s"] > out["remaining_s"]


def test_budget_check_accounts_for_what_is_already_spent():
    almost_exhausted = budget_check(10, spent=999.0, allowance=1000.0)
    fresh = budget_check(10, spent=0.0, allowance=1000.0)
    assert almost_exhausted["remaining_s"] < fresh["remaining_s"]
    assert almost_exhausted["ok"] is False or almost_exhausted["remaining_s"] < 1.0


# ---------------------------------------------------------------------- prompt_token_logprobs


def test_stub_backend_prompt_token_logprobs_default_responder_is_well_formed():
    chat = [{"role": "system", "content": "sys"}, {"role": "user", "content": "one two three"}]
    backend = StubBackend(lambda i, c: "reply", name="stub-plp")

    reads = backend.prompt_token_logprobs([chat])

    assert len(reads) == 1
    read = reads[0]
    assert len(read.token_ids) == len(read.tokens) == len(read.logprobs)
    assert read.logprobs[0] is None
    assert all(lp is not None for lp in read.logprobs[1:])
    assert read.complete is True
    assert read.backend == "stub"


def test_stub_backend_prompt_token_logprobs_is_deterministic():
    chat = [{"role": "user", "content": "a fixed piece of text"}]
    backend = StubBackend(lambda i, c: "reply", name="stub-plp-2")

    first = backend.prompt_token_logprobs([chat])[0]
    second = backend.prompt_token_logprobs([chat])[0]

    assert first.token_ids == second.token_ids
    assert first.logprobs == second.logprobs


def test_stub_backend_prompt_token_logprobs_honours_a_custom_responder():
    def responder(index, chat):
        return (["x", "y", "z"], [None, -1.0, -2.0])

    backend = StubBackend(lambda i, c: "reply", name="stub-plp-3",
                           prompt_logprob_responder=responder)
    read = backend.prompt_token_logprobs([[{"role": "user", "content": "ignored"}]])[0]

    assert read.token_ids == ["x", "y", "z"]
    assert read.logprobs == [None, -1.0, -2.0]
    assert read.complete is True


def test_stub_backend_prompt_token_logprobs_flags_incompleteness():
    def responder(index, chat):
        return (["x", "y", "z"], [None, -1.0, None])

    backend = StubBackend(lambda i, c: "reply", name="stub-plp-4",
                           prompt_logprob_responder=responder)
    read = backend.prompt_token_logprobs([[{"role": "user", "content": "ignored"}]])[0]

    assert read.complete is False


# ------------------------------------------------------------------------------------ dry run


def test_dry_run_end_to_end_writes_rows_and_a_summary(tmp_path):
    rc = main(["--dry-run", "--out-dir", str(tmp_path)])
    assert rc == 0

    jsonl_files = list(tmp_path.glob("surprisal_*.jsonl"))
    assert len(jsonl_files) == 1
    rows = [json.loads(l) for l in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
    assert rows  # at least one (item, condition) row was written

    for row in rows:
        assert row["condition"] in ("R1", "R2", "R3", "R4")
        assert row["complete"] is True  # the stub's default responder always supplies logprobs
        assert row["n_tokens"] > 0
        assert row["mean_nll"] is not None
        assert isinstance(row["per_token_nll"], list)
        assert len(row["per_token_nll"]) == row["n_tokens"]

    summary_path = tmp_path / "surprisal_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["models"] == ["stub"]
    assert "per_model" in summary


def test_dry_run_respects_limit(tmp_path):
    rc = main(["--dry-run", "--limit", "0", "--out-dir", str(tmp_path)])
    assert rc == 0  # limit 0 means "no cap", must not crash or produce nothing


# ------------------------------------------ Backend.generate's temperature/seed override


def test_stub_generate_with_no_sampling_args_is_unchanged():
    """The no-argument call path must be byte-identical to `generate` before `temperature`/
    `seed` existed: the same canned responder, called the same way, same output."""
    chat = [{"role": "user", "content": "hello"}]
    backend = StubBackend(lambda i, c: f"reply-{i}", name="stub-gen-1")

    before = [backend._responder(i, chat) for i, chat in enumerate([chat, chat])]
    after = backend.generate([chat, chat])

    assert after == before == ["reply-0", "reply-1"]


def test_stub_generate_same_temperature_and_seed_reproduces():
    chat = [{"role": "user", "content": "hello"}]
    backend = StubBackend(lambda i, c: "base", name="stub-gen-2")

    first = backend.generate([chat], temperature=0.7, seed=123)
    second = backend.generate([chat], temperature=0.7, seed=123)

    assert first == second


def test_stub_generate_a_different_seed_need_not_reproduce():
    chat = [{"role": "user", "content": "hello"}]
    backend = StubBackend(lambda i, c: "base", name="stub-gen-3")

    a = backend.generate([chat], temperature=0.7, seed=1)
    b = backend.generate([chat], temperature=0.7, seed=2)

    assert a != b


def test_stub_generate_forwards_sampling_kwargs_to_a_responder_that_declares_them():
    """A responder that opts in (declares `temperature`/`seed`, or takes `**kwargs`) receives
    the real values, not the generic marker fallback."""
    seen = []

    def responder(index, chat, *, temperature=None, seed=None):
        seen.append((index, temperature, seed))
        return "ok"

    backend = StubBackend(responder, name="stub-gen-4")
    backend.generate([[{"role": "user", "content": "x"}]], temperature=0.9, seed=42)

    assert seen == [(0, 0.9, 42)]


def test_stub_generate_zero_temperature_with_no_seed_still_uses_the_plain_path():
    """`temperature=0.0` is given explicitly (not omitted) -- greedy, not a degenerate sampling
    call -- and the seed argument alone must not force the marker-fallback branch into treating
    zero temperature as "sampling"."""
    chat = [{"role": "user", "content": "hello"}]
    backend = StubBackend(lambda i, c: "reply", name="stub-gen-5")

    out = backend.generate([chat], temperature=0.0, seed=7)
    # temperature/seed were passed explicitly (not both None), so the marker path is used, but
    # it must reflect exactly the temperature given -- 0.0, not a substituted nonzero default.
    assert out == ["reply [t=0 seed=7]"]


# --------------------------------------------------------------------------------- cleanliness


def test_touched_files_contain_no_stray_control_characters():
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    paths = [here.parent / "pilot" / "surprisal.py", here / "test_surprisal.py"]

    offenders = []
    for path in paths:
        for i, byte in enumerate(path.read_bytes()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((str(path), i, byte))
    assert offenders == []
