"""Tests for pilot.decoding_sweep (E4): per-sample seed derivation, the sampling adapter, the
modal-choice/agreement summary, the budget check, and --dry-run end to end.

Hand-written fixtures only; no GPU or network."""

import json

import pytest

from pilot import config as C
from pilot.data import Entity, Item
from pilot.extract import Rejection
from pilot.repair import build_conditions, build_corpus_index

from pilot.decoding_sweep import (
    _SweepStubResponder,
    _edited_letter_by_condition,
    _load_greedy_stage3,
    _modal_choice,
    _parse_part,
    _refuse_unsupported_temperatures,
    _sample_generate,
    _select_part,
    check_budget,
    derive_sample_seed,
    estimate_cost_s,
    main,
    mcnemar_on_modal_choice,
    run_stage3_sweep,
)
from pilot.models import StubBackend


# --------------------------------------------------------------------- derive_sample_seed


def test_derive_sample_seed_reproducible():
    a = derive_sample_seed(20260820, "item-1", "R1", 0.7, 2)
    b = derive_sample_seed(20260820, "item-1", "R1", 0.7, 2)
    assert a == b


def test_derive_sample_seed_distinct_per_sample_index():
    seeds = {derive_sample_seed(20260820, "item-1", "R1", 0.7, k) for k in range(5)}
    assert len(seeds) == 5


def test_derive_sample_seed_distinct_per_item():
    a = derive_sample_seed(20260820, "item-1", "R1", 0.7, 0)
    b = derive_sample_seed(20260820, "item-2", "R1", 0.7, 0)
    assert a != b


def test_derive_sample_seed_distinct_per_condition():
    a = derive_sample_seed(20260820, "item-1", "R1", 0.7, 0)
    b = derive_sample_seed(20260820, "item-1", "R2", 0.7, 0)
    assert a != b


def test_derive_sample_seed_distinct_per_temperature():
    a = derive_sample_seed(20260820, "item-1", "R1", 0.7, 0)
    b = derive_sample_seed(20260820, "item-1", "R1", 1.0, 0)
    assert a != b


def test_derive_sample_seed_distinct_per_base_seed():
    a = derive_sample_seed(1, "item-1", "R1", 0.7, 0)
    b = derive_sample_seed(2, "item-1", "R1", 0.7, 0)
    assert a != b


# ------------------------------------------------------------------------------- cost estimate


def test_estimate_cost_s_is_a_product_of_its_four_inputs():
    assert estimate_cost_s(1, 1, 1, 1) > 0
    assert estimate_cost_s(2, 5, 3, 2) == pytest.approx(2 * 5 * 3 * 2 * estimate_cost_s(1, 1, 1, 1))
    assert estimate_cost_s(0, 5, 3, 2) == 0


def test_check_budget_refuses_when_estimate_exceeds_remaining(monkeypatch):
    monkeypatch.setattr(
        "pilot.decoding_sweep.cumulative_gpu_seconds", lambda: C.PROJECT_GPU_BUDGET_S - 1
    )
    with pytest.raises(SystemExit):
        check_budget(estimated_s=100.0, label="test")


def test_check_budget_allows_when_estimate_fits(monkeypatch):
    monkeypatch.setattr("pilot.decoding_sweep.cumulative_gpu_seconds", lambda: 0.0)
    check_budget(estimated_s=100.0, label="test")  # must not raise


# --------------------------------------------------------------- the real-backend refusal


def test_refuse_unsupported_temperatures_allows_greedy_against_a_real_backend():
    _refuse_unsupported_temperatures([C.TEMPERATURE], dry_run=False)  # must not raise


def test_refuse_unsupported_temperatures_blocks_nonzero_against_a_real_backend():
    with pytest.raises(SystemExit):
        _refuse_unsupported_temperatures([0.7], dry_run=False)


def test_refuse_unsupported_temperatures_allows_anything_in_dry_run():
    _refuse_unsupported_temperatures([0.7, 1.0], dry_run=True)  # must not raise


# ------------------------------------------------------------------------------- modal choice


def test_modal_choice_picks_the_most_frequent():
    assert _modal_choice(["A", "A", "B"]) == "A"


def test_modal_choice_ties_break_to_first_occurrence():
    assert _modal_choice(["B", "A", "A", "B"]) == "B"


def test_modal_choice_ignores_unreadable_samples():
    assert _modal_choice([None, "C", None, "C"]) == "C"


def test_modal_choice_none_when_every_sample_unreadable():
    assert _modal_choice([None, None]) is None


def test_modal_choice_none_on_empty_input():
    assert _modal_choice([]) is None


# ------------------------------------------------------------------------- _SweepStubResponder


def test_sweep_stub_responder_varies_with_seed():
    chat = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Question: who?\n\nCandidates:\n\nA) Anna\nprofile\n\n"
                                     "B) Bruno\nprofile\n\nC) Clara\nprofile\n\nD) Dawid\nprofile"},
    ]
    responder = _SweepStubResponder()
    seen = set()
    for seed in range(4):
        responder.seed = seed
        seen.add(responder(0, chat))
    assert len(seen) == 4  # four options, four distinct seeds -> four distinct replies


def test_sample_generate_requires_a_stub_responder_for_a_stub_backend():
    backend = StubBackend(lambda i, c: "The answer is A.", name="stub")
    chat = [{"role": "user", "content": "x"}]
    with pytest.raises(TypeError):
        # calling with stub_responder omitted is a caller error, not silently tolerated --
        # _sample_generate requires it as a keyword argument
        _sample_generate(backend, chat, temperature=0.0, seed=1)


def test_sample_generate_uses_the_given_stub_responder_not_a_fresh_one():
    responder = _SweepStubResponder()
    backend = StubBackend(responder, name="stub")
    chat = [
        {"role": "user", "content": "Question: who?\n\nCandidates:\n\nA) Anna\nprofile\n\n"
                                     "B) Bruno\nprofile"},
    ]
    out0 = _sample_generate(backend, chat, temperature=0.0, seed=0, stub_responder=responder)
    out1 = _sample_generate(backend, chat, temperature=0.0, seed=1, stub_responder=responder)
    assert out0 != out1  # seed 0 -> option A, seed 1 -> option B (2 options, seed % 2)


# ------------------------------------------------------------------------------ fixture item


def make_item() -> tuple[Item, Rejection]:
    item = Item(
        item_id="dsweep-1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker.",
                                      "She was born in 1970 in Krakow.", "She directed Blue River."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer.",
                                       "He worked on several feature films."]),
            Entity("Clara Novak", ["Clara Novak is a Czech screenwriter.",
                                    "Her mother was named Maria Elena Novak."]),
            Entity("Dawid Kowalski", ["Dawid Kowalski is a Polish film editor.",
                                       "He edited three documentaries."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B", title="Bruno Kowalski", matched_by="title", attribute="date_of_birth",
        cue="birth", negation_cue="not",
    )
    return item, rejection


def make_kept() -> dict:
    item, rejection = make_item()
    index = build_corpus_index([item])
    conditions = build_conditions(item, rejection, index, choice_letter="A")
    return {item.item_id: (item, rejection, conditions)}


# --------------------------------------------------------------------- _edited_letter_by_condition


def test_edited_letter_by_condition_matches_the_built_conditions():
    kept = make_kept()
    item_id, (item, rejection, conditions) = next(iter(kept.items()))
    edited = _edited_letter_by_condition(item, conditions, rejection.letter)
    assert edited["R0"] is None
    assert edited["R1"] == "B"
    assert edited["R2"] == "B"
    assert edited["R3"] == edited["R4"]
    assert edited["R3"] not in ("A", "B")  # never the gold or the rival


# ------------------------------------------------------------------------------ run_stage3_sweep


def test_run_stage3_sweep_against_a_stub_produces_variation_and_writes_jsonl(tmp_path):
    kept = make_kept()
    responder = _SweepStubResponder()
    backend = StubBackend(responder, name="stub")

    per_temperature, counts = run_stage3_sweep(
        backend, kept, tmp_path, temperatures=[0.0, 0.7], n_samples=4, base_seed=20260820,
        greedy_by_item=None, stub_responder=responder,
    )
    assert counts["n_items"] == 1
    assert counts["greedy_source"] == "computed_from_sweep"
    assert set(per_temperature) == {"0.0", "0.7"}

    path = tmp_path / "decoding_sweep_stage3_stub.jsonl"
    assert path.exists()
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1 * 5 * 2 * 4  # items * conditions * temperatures * n_samples

    block_00 = per_temperature["0.0"]
    # "degenerate_zero_variance" reflects config.TEMPERATURE alone -- true whenever the real
    # backend would be forced to greedy (zero real variance there); the *stub*'s own replies
    # still vary by seed at temperature 0.0 too (see _sample_generate: the stub branch never
    # special-cases temperature), so this flag is a statement about the real backend's
    # behaviour, not a claim that this particular stub run produced identical samples.
    assert block_00["degenerate_zero_variance"] is True
    block_07 = per_temperature["0.7"]
    assert block_07["degenerate_zero_variance"] is False
    # temperature 0.0's own samples became the computed greedy reference (see
    # run_stage3_sweep); that single shared reference is used for *every* temperature's
    # agreement figures, including 0.7's -- one reference item, valid bounded rates, for both.
    for block in (block_00, block_07):
        for cond, agreement in block["agreement_with_greedy"].items():
            assert agreement["n_items_with_reference"] == 1
            assert 0.0 <= agreement["modal_agreement_rate"] <= 1.0
            assert 0.0 <= agreement["sample_agreement_rate"] <= 1.0


def test_run_stage3_sweep_requires_a_stub_responder_for_a_stub_backend(tmp_path):
    kept = make_kept()
    backend = StubBackend(lambda i, c: "The answer is A.", name="stub")
    with pytest.raises(ValueError):
        run_stage3_sweep(
            backend, kept, tmp_path, temperatures=[0.0], n_samples=1, base_seed=1,
            greedy_by_item=None, stub_responder=None,
        )


def test_run_stage3_sweep_uses_a_supplied_greedy_reference(tmp_path):
    kept = make_kept()
    item_id = next(iter(kept))
    responder = _SweepStubResponder()
    backend = StubBackend(responder, name="stub")

    # A reference that disagrees with whatever the stub actually says -- forces
    # modal_agreement_rate to 0 for every condition, proving the supplied reference (not a
    # self-computed one) is what agreement is measured against.
    greedy_by_item = {item_id: {c: "Z" for c in ("R0", "R1", "R2", "R3", "R4")}}
    per_temperature, counts = run_stage3_sweep(
        backend, kept, tmp_path, temperatures=[0.0], n_samples=2, base_seed=1,
        greedy_by_item=greedy_by_item, stub_responder=responder,
    )
    assert counts["greedy_source"] == "loaded_from_file"
    for agreement in per_temperature["0.0"]["agreement_with_greedy"].values():
        assert agreement["modal_agreement_rate"] == 0.0


# --------------------------------------------------------------------------- mcnemar_on_modal_choice


def test_mcnemar_on_modal_choice_reports_all_four_contrasts():
    modal_by_item = {
        "i1": {"R0": "A", "R1": "B", "R2": "A", "R3": "C", "R4": "A"},
    }
    edited_letter_by_item = {
        "i1": {"R0": None, "R1": "B", "R2": "B", "R3": "C", "R4": "C"},
    }
    result = mcnemar_on_modal_choice(modal_by_item, edited_letter_by_item)
    assert set(result) == {"R1_vs_R2", "R3_vs_R4", "R1_vs_R3", "R2_vs_R4"}
    for contrast in result.values():
        assert "p_exact_two_sided" in contrast


# --------------------------------------------------------------------------- _load_greedy_stage3


def test_load_greedy_stage3_reads_item_condition_choice(tmp_path):
    path = tmp_path / "stage3_model.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in [
            {"item_id": "i1", "condition": "R0", "choice": "A"},
            {"item_id": "i1", "condition": "R1", "choice": "B"},
            {"item_id": "i2", "condition": "R0", "choice": None},
        ]),
        encoding="utf-8",
    )
    loaded = _load_greedy_stage3(path)
    assert loaded == {"i1": {"R0": "A", "R1": "B"}, "i2": {"R0": None}}


# --------------------------------------------------------------------------------- --part


def test_parse_part_accepts_i_of_n():
    assert _parse_part("1/2") == (0, 2)


def test_parse_part_rejects_out_of_range():
    with pytest.raises(SystemExit):
        _parse_part("3/2")


def test_select_part_is_deterministic_and_covers_every_id():
    ids = ["x", "w", "y", "z"]
    parts = [_select_part(ids, (i, 2)) for i in range(2)]
    assert sorted(parts[0] + parts[1]) == sorted(ids)
    assert not (set(parts[0]) & set(parts[1]))


# ---------------------------------------------------------------------------------- --dry-run


def test_dry_run_end_to_end_produces_a_summary(tmp_path):
    rc = main(["--dry-run", "--temperatures", "0.0", "0.7", "--n-samples", "2",
               "--limit", "4", "--out-dir", str(tmp_path)])
    assert rc == 0
    summary = json.loads((tmp_path / "decoding_sweep_summary.json").read_text(encoding="utf-8"))
    assert summary["temperatures"] == [0.0, 0.7]
    assert summary["n_samples"] == 2
    assert summary["dry_run"] is True
    assert "stub" in summary["per_model"]
    model_summary = summary["per_model"]["stub"]
    assert "per_temperature" in model_summary
    assert "mcnemar_on_modal_choice" in model_summary
    assert set(model_summary["per_temperature"]) == {"0.0", "0.7"}


def test_main_requires_temperatures_and_n_samples(tmp_path):
    with pytest.raises(SystemExit):
        main(["--dry-run", "--n-samples", "2", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["--dry-run", "--temperatures", "0.0", "--out-dir", str(tmp_path)])


def test_main_requires_from_stage1_unless_dry_run(tmp_path):
    with pytest.raises(SystemExit):
        main(["--temperatures", "0.0", "--n-samples", "1", "--out-dir", str(tmp_path)])


def test_main_requires_a_greedy_reference(tmp_path):
    # temperature 0.7 only, no --greedy-stage3-dir: no reference is ever available
    with pytest.raises(SystemExit):
        main(["--dry-run", "--temperatures", "0.7", "--n-samples", "1", "--out-dir", str(tmp_path)])


def test_main_refuses_nonzero_temperature_against_a_real_backend(tmp_path):
    # Not --dry-run, temperature != config.TEMPERATURE: must be refused before any file I/O --
    # --from-stage1 points at a path that does not exist, and the refusal must still fire first.
    with pytest.raises(SystemExit):
        main(["--temperatures", "0.7", "--n-samples", "1",
              "--from-stage1", str(tmp_path / "does_not_exist.jsonl"),
              "--greedy-stage3-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
