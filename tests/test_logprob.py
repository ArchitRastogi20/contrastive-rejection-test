"""Tests for the forced-choice letter-probability continuous outcome measure: pilot.models'
letter-probe helpers and pilot.run_experiment's continuous-contrast functions.

Hand-written fixtures only; no GPU or network.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from pilot import repair
from pilot.models import (
    StubBackend,
    append_letter_probe,
    letter_read_from_token_logprobs,
    renormalize_letter_logprobs,
    resolve_letter_token_ids,
)
from pilot.run_experiment import (
    _fixture_item_for_gates,
    all_continuous_contrasts,
    bootstrap_ci_mean_diff,
    continuous_contrast,
    delta_from_baseline,
    r0_baseline_diagnostic,
    run_stage3,
)

# --------------------------------------------------------------------------- renormalisation


def test_renormalize_sums_to_one_and_is_correct():
    raw = {"A": -0.1, "B": -1.0, "C": -2.0, "D": -3.0}
    probs = renormalize_letter_logprobs(raw)

    assert set(probs) == set(raw)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-12)
    assert probs["A"] == max(probs.values())  # least negative logprob -> largest share

    m = max(raw.values())
    exps = {k: math.exp(v - m) for k, v in raw.items()}
    total = sum(exps.values())
    expected = {k: v / total for k, v in exps.items()}
    for k in raw:
        assert math.isclose(probs[k], expected[k], rel_tol=1e-12)


def test_renormalize_ignores_missing_entries_rather_than_imputing():
    raw = {"A": -0.1, "B": None, "C": -2.0}
    probs = renormalize_letter_logprobs(raw)

    assert set(probs) == {"A", "C"}  # B never appears with a made-up value
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-12)


def test_renormalize_of_nothing_found_is_empty():
    assert renormalize_letter_logprobs({"A": None, "B": None}) == {}


# --------------------------------------------------------------------- option count not fixed


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7])
def test_option_count_is_read_from_the_candidates_not_hardcoded(n):
    """The design explicitly forbids hardcoding four candidates -- items can have any number
    of options, and the read must size itself off the item, not off a constant."""
    letters = [chr(ord("A") + i) for i in range(n)]
    token_logprobs = {letter: -float(i) for i, letter in enumerate(letters)}

    read = letter_read_from_token_logprobs(token_logprobs, letters, backend="vllm")

    assert read.candidates == letters
    assert len(read.candidates) == n
    assert read.complete
    assert set(read.probs) == set(letters)
    assert math.isclose(sum(read.probs.values()), 1.0, rel_tol=1e-9)


# --------------------------------------------------------- missing letter: flagged, excluded


def test_missing_letter_in_topk_is_flagged_not_imputed():
    # Only A, B, C showed up in this (simulated) vLLM top-k; D never did.
    token_logprobs = {"A": -0.2, "B": -1.5, "C": -3.0}
    read = letter_read_from_token_logprobs(token_logprobs, ["A", "B", "C", "D"], backend="vllm")

    assert read.raw_logprobs["D"] is None
    assert read.complete is False
    assert "D" not in read.probs  # renormalised only over what was found -- no imputation
    assert math.isclose(sum(read.probs.values()), 1.0, rel_tol=1e-9)


def test_a_letter_appearing_as_two_distinct_tokens_has_its_mass_combined():
    # "A" and " A" (leading-space variant) both decode to the letter A; both are genuine ways
    # the model could have emitted it, so neither is discarded in favour of the other.
    token_logprobs = {"A": -2.0, " A": -2.0, "B": -1.0}
    read = letter_read_from_token_logprobs(token_logprobs, ["A", "B"], backend="vllm")

    assert read.complete
    assert read.raw_logprobs["A"] > -2.0  # combined mass exceeds either single entry alone


# ---------------------------------------------------------- leading-space tokenization (HF path)


class _FakeTokenizer:
    """A minimal stand-in for a real tokenizer, enough to exercise
    `resolve_letter_token_ids`'s empirical resolution without a BPE model: a letter directly
    preceded by a space merges with that space into one "leading-space" token (mimicking a
    sentencepiece/BPE vocabulary's very common merged " X" token); anything else is one bare
    token per character.
    """

    def __call__(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == " " and i + 1 < len(text) and text[i + 1].isalpha():
                ids.append(1000 + ord(text[i + 1]))  # merged leading-space token
                i += 2
                continue
            if ch.isalpha():
                ids.append(2000 + ord(ch))  # bare token
                i += 1
                continue
            ids.append(ord(ch))
            i += 1
        return SimpleNamespace(input_ids=ids)

    def convert_ids_to_tokens(self, ids):
        out = []
        for tid in ids:
            if tid >= 2000:
                out.append(chr(tid - 2000))
            elif tid >= 1000:
                out.append("▁" + chr(tid - 1000))  # sentencepiece leading-space marker
            else:
                out.append(chr(tid))
        return out


def test_leading_space_variant_resolved_when_the_prompt_ends_on_a_space():
    tok = _FakeTokenizer()
    prompt = "Question ends with a colon, answer: "  # ends on a literal space

    token_ids, variant = resolve_letter_token_ids(tok, prompt, ["A", "B"])

    assert variant == {"A": "leading_space", "B": "leading_space"}
    assert token_ids == {"A": 1000 + ord("A"), "B": 1000 + ord("B")}


def test_bare_variant_resolved_when_the_prompt_ends_without_a_space():
    tok = _FakeTokenizer()
    prompt = "Question ends with a colon, answer:\n"  # ends on a newline, not a space

    token_ids, variant = resolve_letter_token_ids(tok, prompt, ["A", "B"])

    assert variant == {"A": "bare", "B": "bare"}
    assert token_ids == {"A": 2000 + ord("A"), "B": 2000 + ord("B")}


# --------------------------------------------------------------------------- bootstrap CI


def test_bootstrap_ci_is_deterministic_under_a_fixed_seed():
    diffs = [0.1, 0.3, -0.05, 0.2, 0.15, 0.0, 0.25, -0.1, 0.05, 0.4]
    first = bootstrap_ci_mean_diff(diffs, seed=12345, n_resamples=500)
    second = bootstrap_ci_mean_diff(diffs, seed=12345, n_resamples=500)

    assert first == second  # byte-identical, not merely close


def test_bootstrap_ci_with_a_different_seed_need_not_match():
    diffs = [0.1, 0.3, -0.05, 0.2, 0.15, 0.0, 0.25, -0.1, 0.05, 0.4]
    a = bootstrap_ci_mean_diff(diffs, seed=1, n_resamples=500)
    b = bootstrap_ci_mean_diff(diffs, seed=2, n_resamples=500)

    assert a["mean"] == b["mean"]  # same data, same point estimate
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


def test_bootstrap_ci_brackets_a_known_mean_on_a_synthetic_sample():
    diffs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # mean exactly 0.5
    result = bootstrap_ci_mean_diff(diffs, seed=42, n_resamples=2000)

    assert math.isclose(result["mean"], 0.5, rel_tol=1e-12)
    assert result["ci_low"] < 0.5 < result["ci_high"]  # a real bracket, not a point
    assert result["seed"] == 42
    assert result["n"] == len(diffs)


def test_bootstrap_ci_records_the_seed_and_handles_the_empty_case():
    result = bootstrap_ci_mean_diff([], seed=7)
    assert result == {
        "mean": None, "ci_low": None, "ci_high": None, "n": 0, "seed": 7, "n_resamples": 10000,
    }


def test_all_continuous_contrasts_reports_both_measures_for_every_contrast():
    """Both the primary (delta-corrected) and the raw quantity must be reported, per the
    request to keep the difference between them inspectable rather than hiding one."""
    prob_outcomes = {
        "item-1": {
            "R1": {"p_edited": 0.6, "complete": True, "r0_complete": True, "delta_p_edited": 0.1},
            "R2": {"p_edited": 0.2, "complete": True, "r0_complete": True, "delta_p_edited": -0.1},
            "R3": {"p_edited": 0.5, "complete": True, "r0_complete": True, "delta_p_edited": 0.3},
            "R4": {"p_edited": 0.1, "complete": True, "r0_complete": True, "delta_p_edited": 0.0},
        },
    }
    out = all_continuous_contrasts(prob_outcomes, seed=1)

    assert set(out) == {"delta_p_edited", "p_edited_raw"}
    expected_contrasts = {"R1_vs_R2", "R3_vs_R4", "R1_vs_R3", "R2_vs_R4"}
    assert set(out["delta_p_edited"]) == expected_contrasts
    assert set(out["p_edited_raw"]) == expected_contrasts
    for contrast in expected_contrasts:
        assert out["delta_p_edited"][contrast]["measure"] == "delta_p_edited"
        assert out["p_edited_raw"][contrast]["measure"] == "p_edited"


def test_continuous_contrast_excludes_incomplete_reads_rather_than_imputing():
    prob_outcomes = {
        "item-1": {
            "R1": {"p_edited": 0.6, "complete": True},
            "R2": {"p_edited": 0.2, "complete": True},
        },
        "item-2": {  # R2's read is incomplete: this pair is excluded, not imputed
            "R1": {"p_edited": 0.5, "complete": True},
            "R2": {"p_edited": None, "complete": False},
        },
        "item-3": {
            "R1": {"p_edited": 0.4, "complete": True},
            "R2": {"p_edited": 0.1, "complete": True},
        },
    }
    result = continuous_contrast(prob_outcomes, "R1", "R2", seed=1, measure="p_edited")

    assert result["n_pairs_total"] == 3
    assert result["n_pairs_complete"] == 2
    assert result["n_excluded_incomplete_condition"] == 1
    assert math.isclose(result["mean"], ((0.6 - 0.2) + (0.4 - 0.1)) / 2, rel_tol=1e-12)
    assert result["contrast"] == "R1-R2"
    assert result["measure"] == "p_edited"


def test_continuous_contrast_defaults_to_delta_p_edited():
    """The four official contrasts must use the baseline-corrected quantity, not raw
    `p_edited`, unless a caller explicitly asks for the raw one -- so the default matters."""
    prob_outcomes = {
        "item-1": {
            "R1": {"p_edited": 0.6, "complete": True, "r0_complete": True, "delta_p_edited": 0.05},
            "R2": {"p_edited": 0.2, "complete": True, "r0_complete": True, "delta_p_edited": -0.10},
        },
    }
    result = continuous_contrast(prob_outcomes, "R1", "R2", seed=1)  # no `measure` given

    assert result["measure"] == "delta_p_edited"
    assert math.isclose(result["mean"], 0.05 - (-0.10), rel_tol=1e-12)


def test_continuous_contrast_excludes_and_counts_an_incomplete_r0_baseline():
    """A condition's own read can be complete while the item's R0 baseline is not -- that must
    still exclude the pair from the delta-based analysis, and be counted separately from a
    plain incomplete-condition exclusion so the two reasons stay distinguishable."""
    prob_outcomes = {
        "item-1": {
            "R1": {"p_edited": 0.6, "complete": True, "r0_complete": True, "delta_p_edited": 0.1},
            "R2": {"p_edited": 0.2, "complete": True, "r0_complete": True, "delta_p_edited": -0.05},
        },
        "item-2": {  # both conditions' own reads are complete, but R0 was not
            "R1": {"p_edited": 0.5, "complete": True, "r0_complete": False, "delta_p_edited": None},
            "R2": {"p_edited": 0.3, "complete": True, "r0_complete": False, "delta_p_edited": None},
        },
    }
    result = continuous_contrast(prob_outcomes, "R1", "R2", seed=1)

    assert result["n_pairs_total"] == 2
    assert result["n_pairs_complete"] == 1
    assert result["n_excluded_incomplete_r0"] == 1
    assert result["n_excluded_incomplete_condition"] == 0


# ------------------------------------------------------------- delta_from_baseline (pure function)


def test_delta_uses_the_same_letter_on_both_sides_not_a_different_one():
    condition_probs = {"A": 0.7, "B": 0.2, "C": 0.1}
    baseline_probs = {"A": 0.1, "B": 0.6, "C": 0.3}

    # If the delta accidentally compared against a different letter, this would come out
    # differently for each of A/B/C -- pin all three so a mix-up cannot pass silently.
    assert math.isclose(delta_from_baseline(condition_probs, baseline_probs, "A"),
                         0.7 - 0.1, rel_tol=1e-12)
    assert math.isclose(delta_from_baseline(condition_probs, baseline_probs, "B"),
                         0.2 - 0.6, rel_tol=1e-12)
    assert math.isclose(delta_from_baseline(condition_probs, baseline_probs, "C"),
                         0.1 - 0.3, rel_tol=1e-12)
    # explicitly not equal to a same-side comparison against the wrong letter
    assert delta_from_baseline(condition_probs, baseline_probs, "A") != (
        condition_probs["A"] - baseline_probs["B"]
    )


def test_delta_is_none_when_the_letter_is_missing_on_either_side():
    assert delta_from_baseline({"A": 0.5}, {"A": 0.4}, None) is None  # no target letter (R0)
    assert delta_from_baseline({"B": 0.5}, {"A": 0.4}, "A") is None  # missing from condition
    assert delta_from_baseline({"A": 0.5}, {"B": 0.4}, "A") is None  # missing from baseline


@pytest.mark.parametrize("letter", ["A", "B", "C"])
def test_delta_against_its_own_baseline_is_zero_by_construction(letter):
    """R0 compared against itself, on any letter, must be exactly zero -- R0 is the baseline
    every other condition's delta is measured against, so this is the sanity floor: if the
    baseline-vs-baseline delta were ever nonzero, the whole correction would be untrustworthy."""
    probs = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert delta_from_baseline(probs, probs, letter) == 0.0


# ------------------------------------------------------ raw vs. delta can disagree in sign


def test_raw_p_edited_and_delta_p_edited_comparisons_can_disagree_in_sign():
    """Deliberately chosen so the raw P(edited) comparison and the baseline-corrected delta
    comparison point in opposite directions -- this is exactly the confound `delta_from_baseline`
    exists to correct: R1/R2's target (the rival) and R3/R4's target (a third option) do not
    start from the same R0 baseline, so a raw comparison across them can be dominated by that
    baseline gap rather than by anything the edit did. A regression that quietly switched the
    official contrasts back to the raw quantity must not pass silently: this pins both numbers
    and their disagreement."""
    prob_outcomes = {
        "item-1": {
            "R1": {  # rival started high (0.8) and barely moved
                "p_edited": 0.9, "complete": True, "r0_complete": True, "delta_p_edited": 0.9 - 0.8,
            },
            "R3": {  # third option started low (0.1) and moved a lot
                "p_edited": 0.5, "complete": True, "r0_complete": True, "delta_p_edited": 0.5 - 0.1,
            },
        },
    }
    raw = continuous_contrast(prob_outcomes, "R1", "R3", seed=1, measure="p_edited")
    delta = continuous_contrast(prob_outcomes, "R1", "R3", seed=1, measure="delta_p_edited")

    assert math.isclose(raw["mean"], 0.9 - 0.5, rel_tol=1e-12)      # raw: R1 looks bigger
    assert math.isclose(delta["mean"], 0.1 - 0.4, rel_tol=1e-12)    # delta: R3 actually moved more
    assert (raw["mean"] > 0) != (delta["mean"] > 0)  # the sign disagreement is the point


# ----------------------------------------------------------------- R0 baseline diagnostic


def test_r0_baseline_diagnostic_reports_each_targets_own_range_without_double_counting():
    prob_outcomes = {
        "item-1": {
            "R1": {"r0_p_target": 0.8}, "R2": {"r0_p_target": 0.8},  # same target as R1
            "R3": {"r0_p_target": 0.1}, "R4": {"r0_p_target": 0.1},  # same target as R3
        },
        "item-2": {
            "R1": {"r0_p_target": 0.6}, "R2": {"r0_p_target": 0.6},
            "R3": {"r0_p_target": 0.3}, "R4": {"r0_p_target": 0.3},
        },
    }
    out = r0_baseline_diagnostic(prob_outcomes)

    # only R1/R3 are read, once per item -- R2/R4 sharing the same target must not double it
    assert out["rival_target_R1_R2"]["n"] == 2
    assert math.isclose(out["rival_target_R1_R2"]["mean"], 0.7, rel_tol=1e-12)
    assert out["rival_target_R1_R2"]["min"] == 0.6
    assert out["rival_target_R1_R2"]["max"] == 0.8
    assert out["third_option_target_R3_R4"]["n"] == 2
    assert math.isclose(out["third_option_target_R3_R4"]["mean"], 0.2, rel_tol=1e-12)


def test_r0_baseline_diagnostic_on_no_data_is_all_none():
    out = r0_baseline_diagnostic({})
    assert out["rival_target_R1_R2"] == {"n": 0, "mean": None, "min": None, "max": None}


# ------------------------------------------------------- append_letter_probe leaves the context alone


def test_append_letter_probe_only_adds_one_turn():
    chat = [{"role": "system", "content": "sys"}, {"role": "user", "content": "user text"}]
    probed = append_letter_probe(chat)

    # Same number of turns: the instruction is folded into the existing final user turn, not
    # appended as a new one -- two consecutive user turns with no assistant reply between them
    # breaks strict alternation, which some chat templates (Mistral-7B-Instruct-v0.3) enforce.
    assert len(probed) == len(chat) == 2
    assert probed[0] == chat[0]  # the system turn is untouched
    assert probed[1]["role"] == "user"
    assert probed[1]["content"].startswith(chat[1]["content"])
    assert "single letter" in probed[1]["content"]


def test_append_letter_probe_rejects_non_user_final_turn():
    chat = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "reply"}]
    with pytest.raises(ValueError):
        append_letter_probe(chat)


# ------------------------------------------------------------- stub backend: both measures, end to end


def _build_kept_fixture():
    item, rejection = _fixture_item_for_gates()
    index = repair.build_corpus_index([item])
    conditions = repair.build_conditions(item, rejection, index, choice_letter="A")
    return item, rejection, {item.item_id: (item, rejection, conditions)}


class _Scripted:
    """Cycles through fixed replies, one per `generate()` call -- deterministic stand-in for
    a model, same convention `test_repair.py` uses."""

    def __init__(self, letters):
        self.letters = list(letters)

    def __call__(self, _index, _chat):
        return f"The answer is {self.letters.pop(0)}."


def test_stub_backend_path_produces_a_complete_record_with_both_measures(tmp_path):
    _, _, kept = _build_kept_fixture()
    backend = StubBackend(_Scripted(["A", "B", "A", "C", "D"]), name="scripted-lp")

    outcomes, counts = run_stage3(backend, kept, tmp_path)

    assert counts["n_items"] == 1
    rows = [
        json.loads(l)
        for l in (tmp_path / "stage3_scripted-lp.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 5
    for row in rows:
        assert row["choice"] is not None  # the free-text measure is present
        assert "chosen_is_repaired_rival" in row

        probe = row["letter_probe"]  # the new, additional measure is present too
        assert probe["complete"] is True
        assert set(probe["candidates"]) == set("ABCD")
        assert all(v is not None for v in probe["raw_logprobs"].values())
        assert math.isclose(sum(probe["probs"].values()), 1.0, rel_tol=1e-9)
        assert probe["backend"] == "stub"

    item_id = next(iter(kept))
    assert set(counts["prob_outcomes"][item_id]) == {"R0", "R1", "R2", "R3", "R4"}
    for label in ("R1", "R2", "R3", "R4"):
        entry = counts["prob_outcomes"][item_id][label]
        assert entry["complete"] is True
        assert entry["r0_complete"] is True
        assert entry["p_edited"] is not None
        # the default stub responder is context-independent, so R0's read and every other
        # condition's read are identical -- the delta against R0 on the same letter is 0
        assert entry["r0_p_target"] is not None
        assert math.isclose(entry["delta_p_edited"], 0.0, abs_tol=1e-12)


def test_stub_backend_records_an_incomplete_read_when_a_letter_is_missing(tmp_path):
    """A `letter_prob_responder` that omits a candidate produces `complete: False` and a
    `None` raw logprob for it -- the wiring end to end, not just the pure function above."""
    _, _, kept = _build_kept_fixture()

    def incomplete_responder(_index, _chat, candidates):
        return {letter: -float(i) for i, letter in enumerate(candidates) if letter != "D"}

    backend = StubBackend(
        _Scripted(["A", "B", "A", "C", "D"]), name="incomplete-lp",
        letter_prob_responder=incomplete_responder,
    )
    outcomes, counts = run_stage3(backend, kept, tmp_path)

    rows = [
        json.loads(l)
        for l in (tmp_path / "stage3_incomplete-lp.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        probe = row["letter_probe"]
        assert probe["complete"] is False
        assert probe["raw_logprobs"]["D"] is None
        assert "D" not in probe["probs"]

    item_id = next(iter(kept))
    for label in ("R1", "R2", "R3", "R4"):
        entry = counts["prob_outcomes"][item_id][label]
        assert entry["complete"] is False
        assert entry["delta_p_edited"] is None  # excluded, never imputed
        assert entry["r0_p_target"] is None

    # An incomplete read must not contribute a paired diff to the continuous analysis.
    result = continuous_contrast(counts["prob_outcomes"], "R1", "R2", seed=1)
    assert result["n_pairs_complete"] == 0


def test_stub_backend_excludes_and_counts_pairs_whose_r0_baseline_is_incomplete(tmp_path):
    """Only R0's read is incomplete here (a responder that fails just once, on the very first
    call -- R0 is always processed first for a given item). Every other condition's own read
    is complete, but the delta they need R0 for cannot be trusted, so it must be excluded and
    counted as an R0-incomplete exclusion specifically, not folded into a generic one."""
    _, _, kept = _build_kept_fixture()
    calls = {"n": 0}

    def r0_only_incomplete(_index, _chat, candidates):
        calls["n"] += 1
        if calls["n"] == 1:  # R0
            return {letter: -float(i) for i, letter in enumerate(candidates) if letter != "D"}
        return {letter: -float(i) for i, letter in enumerate(candidates)}

    backend = StubBackend(
        _Scripted(["A", "B", "A", "C", "D"]), name="r0-incomplete-lp",
        letter_prob_responder=r0_only_incomplete,
    )
    outcomes, counts = run_stage3(backend, kept, tmp_path)

    item_id = next(iter(kept))
    probs = counts["prob_outcomes"][item_id]
    assert probs["R0"]["complete"] is False
    for label in ("R1", "R2", "R3", "R4"):
        assert probs[label]["complete"] is True         # this condition's own read is fine
        assert probs[label]["r0_complete"] is False      # but the baseline it needs is not
        assert probs[label]["delta_p_edited"] is None    # excluded, not imputed
        assert probs[label]["r0_p_target"] is None
        assert probs[label]["p_edited"] is not None       # the raw quantity is unaffected

    result = continuous_contrast(counts["prob_outcomes"], "R1", "R2", seed=1)
    assert result["n_pairs_complete"] == 0
    assert result["n_excluded_incomplete_r0"] == 1
    assert result["n_excluded_incomplete_condition"] == 0


# --------------------------------------------------- the argmax fields are unchanged by this


def test_argmax_fields_match_the_pre_existing_deterministic_rule(tmp_path):
    """Same fixture and scripted replies as `test_repair.py`'s own
    `test_stage3_records_the_edited_letter_and_its_own_correctness_per_condition`: rival B,
    R3/R4 target C. A second, independent check (in this file, on this task's own fixture)
    that adding the letter-probe call did not disturb `choice` / `edited_letter` /
    `chosen_is_edited` / `chosen_is_repaired_rival` -- the pre-existing argmax measure.
    """
    _, rejection, kept = _build_kept_fixture()
    assert rejection.letter == "B"
    backend = StubBackend(_Scripted(["A", "B", "A", "C", "D"]), name="argmax-check")

    outcomes, counts = run_stage3(backend, kept, tmp_path)

    rows = [
        json.loads(l)
        for l in (tmp_path / "stage3_argmax-check.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_condition = {r["condition"]: r for r in rows}

    expected = {
        # condition: (choice, edited_letter, chosen_is_edited, chosen_is_repaired_rival)
        "R0": ("A", None, None, False),
        "R1": ("B", "B", True, True),
        "R2": ("A", "B", False, False),
        "R3": ("C", "C", True, False),
        "R4": ("D", "C", False, False),
    }
    for cond, (choice, edited_letter, chosen_is_edited, chosen_is_repaired_rival) in expected.items():
        row = by_condition[cond]
        assert row["choice"] == choice, cond
        assert row["edited_letter"] == edited_letter, cond
        assert row["chosen_is_edited"] == chosen_is_edited, cond
        assert row["chosen_is_repaired_rival"] == chosen_is_repaired_rival, cond
        assert row["rival_letter"] == "B", cond


# --------------------------------------------------------------- no stray control characters


def test_touched_files_contain_no_stray_control_characters():
    """A patch once wrote a literal backspace where the regex needed a word boundary. Cheap to
    check for, invisible to read for -- see test_repair.py / test_extract.py / test_r3_recency.py's
    identical guard, scoped here to just the files this task touched or added."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    paths = [here.parent / "pilot" / "models.py", here.parent / "pilot" / "run_experiment.py",
             here / "test_logprob.py"]

    offenders = []
    for path in paths:
        for i, byte in enumerate(path.read_bytes()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((str(path), i, byte))
    assert offenders == []
