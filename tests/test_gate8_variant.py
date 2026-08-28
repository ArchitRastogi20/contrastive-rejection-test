"""Tests for harness.gate8_variant (E3): the R3/R4 target-selection strategies, and the script's
own plumbing (budget check, --part sharding, per-item seeding, --dry-run end to end).

Hand-written fixtures only; no GPU or network."""

import copy
import json
import random

import pytest

from harness import config as C
from harness.data import Entity, Item
from harness.extract import Rejection
from harness.repair import (
    R3_STRATEGIES,
    RepairUnavailable,
    build_conditions_with_diagnostics,
    build_corpus_index,
    check_integrity,
    select_r3_target_index,
)

from harness.gate8_variant import (
    _item_rng,
    _parse_part,
    _select_part,
    check_budget,
    estimate_cost_s,
    main,
    run_stage2_variant,
)


# --------------------------------------------------------------- select_r3_target_index itself


def test_prefer_lacking_picks_first_candidate_that_lacks_the_attribute():
    # candidates [3,4,5]; 3 and 5 lack the attribute, 4 carries it
    idx = select_r3_target_index([3, 4, 5], [3, 5], [4], n_options=6, strategy="prefer_lacking")
    assert idx == 3


def test_prefer_having_picks_first_candidate_that_carries_the_attribute():
    idx = select_r3_target_index([3, 4, 5], [3, 5], [4], n_options=6, strategy="prefer_having")
    assert idx == 4


def test_prefer_having_falls_back_to_first_available_when_none_qualifies():
    # no candidate carries the attribute -- prefer_having has nothing to prefer
    idx = select_r3_target_index([3, 4, 5], [3, 4, 5], [], n_options=6, strategy="prefer_having")
    assert idx == 3


def test_prefer_lacking_falls_back_to_first_available_at_n_options_4():
    # the preference is confined to n_options > 4 (config.PREFER_LACKING_TARGET_ABOVE); at 4
    # it must behave exactly as it always has: first available, ignoring the attribute split.
    idx = select_r3_target_index([2, 3], [3], [2], n_options=4, strategy="prefer_lacking")
    assert idx == 2


def test_random_strategy_picks_from_the_candidate_list():
    rng = random.Random(1)
    idx = select_r3_target_index([3, 4, 5], [3, 5], [4], n_options=6, strategy="random", rng=rng)
    assert idx in (3, 4, 5)


def test_random_strategy_requires_an_rng():
    with pytest.raises(ValueError):
        select_r3_target_index([3, 4, 5], [3, 5], [4], n_options=6, strategy="random", rng=None)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        select_r3_target_index([3, 4, 5], [3, 5], [4], n_options=6, strategy="bogus")


def test_r3_strategies_constant_lists_exactly_the_three_named_rules():
    assert set(R3_STRATEGIES) == {"prefer_lacking", "prefer_having", "random"}


# --------------------------------------------------------------------- _item_rng reproducibility


def test_item_rng_same_seed_same_item_reproduces():
    r1 = _item_rng(20260820, "item-A")
    r2 = _item_rng(20260820, "item-A")
    assert r1.random() == r2.random()
    assert r1.randrange(1000) == r2.randrange(1000)


def test_item_rng_different_items_diverge():
    r1 = _item_rng(20260820, "item-A")
    r2 = _item_rng(20260820, "item-B")
    assert r1.random() != r2.random()


def test_item_rng_different_seeds_diverge_for_the_same_item():
    r1 = _item_rng(1, "item-A")
    r2 = _item_rng(2, "item-A")
    assert r1.random() != r2.random()


# ------------------------------------------------------------------------------ six-option fixture


def make_six_option_item() -> tuple[Item, Rejection]:
    """Gold Anna (A), rival Bruno (B, missing date_of_birth), the model's chosen distractor
    Clara (C, carries date_of_birth). Non-excluded candidates for R3/R4: Dawid (D) and Feliks
    (F) lack date_of_birth; Elena (E) carries it -- giving all three strategies a genuine,
    distinct choice among three candidates (matching run 2's own six-option fixture in
    test_repair.py, extended by one more "has the attribute" option)."""
    item = Item(
        item_id="gate8-six-1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker.",
                                      "She was born in 1970 in Krakow.",
                                      "She directed Blue River."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer.",
                                       "He worked on several feature films."]),
            Entity("Clara Novak", ["Clara Novak was born in 1925.",
                                    "Clara Novak worked as an editor."]),
            Entity("Dawid Kowalski", ["Dawid Kowalski is a Polish film editor.",
                                       "He edited three documentaries."]),
            Entity("Elena Sikorska", ["Elena Sikorska was born in 1931."]),
            Entity("Feliks Nowak", ["Feliks Nowak worked as a translator."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B", title="Bruno Kowalski", matched_by="title", attribute="date_of_birth",
        cue="birth", negation_cue="never",
    )
    return item, rejection


# -------------------------------------------------------------- prefer_lacking is unchanged


def test_prefer_lacking_default_matches_todays_known_value():
    """`test_repair.py::test_six_option_item_prefers_the_candidate_that_lacks_the_attribute`
    already pins this fixture shape's expected pick to "E" (index 4). This mirrors that pin
    with gate8_variant's own fixture, so a regression in the shared default path is caught here
    too, not only in test_repair.py."""
    item, rejection = make_six_option_item()
    index = build_corpus_index([item])
    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="C")
    assert check_integrity(conditions, item, rejection) == []
    assert diag.r3_picked_letter == "D"
    assert diag.r3_strategy == "prefer_lacking"


def test_explicit_prefer_lacking_matches_the_implicit_default():
    item, rejection = make_six_option_item()
    index = build_corpus_index([item])
    default_conditions, default_diag = build_conditions_with_diagnostics(
        item, rejection, index, choice_letter="C"
    )
    explicit_conditions, explicit_diag = build_conditions_with_diagnostics(
        item, rejection, index, choice_letter="C", r3_strategy="prefer_lacking"
    )
    assert default_diag == explicit_diag
    for label in ("R0", "R1", "R2", "R3", "R4"):
        assert [o.sentences for o in default_conditions[label].options] == [
            o.sentences for o in explicit_conditions[label].options
        ]


def test_prefer_having_builds_the_true_counterfactual_candidate_then_fails_gate8():
    """Elena (index 4, letter E) is the only non-excluded candidate that already carries
    date_of_birth, so prefer_having's own selection rule (tested directly, in isolation, by
    `test_prefer_having_picks_the_first_candidate_that_carries_the_attribute`) must land there.
    Gate 8 then fails for *every* R1/R2 combination -- the target already carries the attribute
    before any edit, so the post-edit profile does too, regardless of which sentence R4
    inserts -- so `build_conditions_with_diagnostics` exhausts its whole search and raises,
    exactly as it does for any other item where no candidate can pass every gate. This is the
    end-to-end confirmation that the true counterfactual arm behaves as the module docstring
    says: expressible, but not expected to build items."""
    item, rejection = make_six_option_item()
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable) as exc_info:
        build_conditions_with_diagnostics(
            item, rejection, index, choice_letter="C", r3_strategy="prefer_having"
        )
    exc = exc_info.value
    assert exc.gate_failures is not None
    assert any(f.startswith("gate8:") for f in exc.gate_failures)


def test_random_strategy_is_reproducible_and_records_its_own_letter():
    item, rejection = make_six_option_item()
    index = build_corpus_index([item])
    rng1 = _item_rng(20260820, item.item_id)
    rng2 = _item_rng(20260820, item.item_id)
    conditions1, diag1 = build_conditions_with_diagnostics(
        item, rejection, index, choice_letter="C", r3_strategy="random", r3_rng=rng1
    )
    conditions2, diag2 = build_conditions_with_diagnostics(
        item, rejection, index, choice_letter="C", r3_strategy="random", r3_rng=rng2
    )
    assert diag1 == diag2
    assert diag1.r3_strategy == "random"
    assert diag1.r3_picked_letter in ("D", "E", "F")
    for label in ("R0", "R1", "R2", "R3", "R4"):
        assert [o.sentences for o in conditions1[label].options] == [
            o.sentences for o in conditions2[label].options
        ]


# ---------------------------------------------------------------- all eight gates, any strategy


def _gate_fires(conditions_broken: dict, item: Item, rejection: Rejection, prefix: str) -> bool:
    failures = check_integrity(conditions_broken, item, rejection)
    return any(f.startswith(prefix) for f in failures)


def test_all_eight_gates_still_fire_under_the_random_strategy():
    """The eight integrity gates are a property of the built Items, not of which strategy chose
    the R3/R4 target -- deliberately break each one, on conditions built under strategy=random,
    and confirm `check_integrity` still catches every one of them, exactly as it does under the
    default strategy (see test_repair.py's own gate battery)."""
    item, rejection = make_six_option_item()
    index = build_corpus_index([item])
    rng = _item_rng(999, item.item_id)
    conditions, diag = build_conditions_with_diagnostics(
        item, rejection, index, choice_letter="C", r3_strategy="random", r3_rng=rng
    )
    assert check_integrity(conditions, item, rejection) == []
    r3_idx = ord(diag.r3_picked_letter) - ord("A")

    # gate 1: undo the R1 edit so the attribute is still absent
    broken = copy.deepcopy(conditions)
    broken["R1"].options[1].sentences.pop()
    assert _gate_fires(broken, item, rejection, "gate1:")

    # gate 2: make R2 also repair the named attribute
    broken = copy.deepcopy(conditions)
    broken["R2"].options[1].sentences.append("Bruno Kowalski was born in 1970 in Krakow.")
    assert _gate_fires(broken, item, rejection, "gate2:")

    # gate 3: make R2's inserted sentence far longer than R1's
    broken = copy.deepcopy(conditions)
    broken["R2"].options[1].sentences[-1] = broken["R2"].options[1].sentences[-1] + " " + "word " * 40
    assert _gate_fires(broken, item, rejection, "gate3:")

    # gate 4: insert a date of birth after an existing date of death
    broken = copy.deepcopy(conditions)
    broken["R1"].options[1].sentences[-1] = "Bruno Kowalski was born in 1999."
    broken["R1"].options[1].sentences.append("Bruno Kowalski died in 1950.")
    assert _gate_fires(broken, item, rejection, "gate4:")

    # gate 5: the inserted sentence names another candidate
    broken = copy.deepcopy(conditions)
    broken["R1"].options[1].sentences[-1] = "Bruno Kowalski was born the same year as Clara Novak."
    assert _gate_fires(broken, item, rejection, "gate5:")

    # gate 6: touch the gold profile
    broken = copy.deepcopy(conditions)
    broken["R1"].options[0].sentences.append("This sentence should never be here.")
    assert _gate_fires(broken, item, rejection, "gate6:")

    # gate 7: reorder the options in one condition
    broken = copy.deepcopy(conditions)
    broken["R1"].options[0], broken["R1"].options[1] = broken["R1"].options[1], broken["R1"].options[0]
    assert _gate_fires(broken, item, rejection, "gate7:")

    # gate 8: make R4 also repair the named attribute, at whichever option random picked
    broken = copy.deepcopy(conditions)
    broken["R4"].options[r3_idx].sentences.append(
        f"{item.options[r3_idx].title} was born in 1970 in Krakow."
    )
    assert _gate_fires(broken, item, rejection, "gate8:")


def test_forced_n_options_4_case_is_untouched_by_any_strategy():
    """At n_options<=4 the preference is disabled for every strategy except "random" (which
    ignores n_options entirely by design) -- prefer_lacking and prefer_having must both fall
    back to the historical "first available" rule, exactly reproducing runs 1-2."""
    item = Item(
        item_id="forced1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker.",
                                      "She was born in 1970 in Krakow.", "She directed Blue River."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
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
    index = build_corpus_index([item])
    for strategy in ("prefer_lacking", "prefer_having"):
        conditions, diag = build_conditions_with_diagnostics(
            item, rejection, index, choice_letter="C", r3_strategy=strategy
        )
        assert diag.r3_picked_letter == "D"
        assert check_integrity(conditions, item, rejection) == []


# ---------------------------------------------------------------------- run_stage2_variant


def _stage1_row_via_stub(item: Item) -> dict:
    """A genuine stage1_<model>.jsonl row, produced the same way `run_experiment.run_stage1`
    (and `gate8_variant.main --dry-run`) produce one: render the prompt, run it through the
    project's own scripted stub (`run_pilot._StubReplies`), extract the row with `_stage1_row`.
    Hand-writing response text that the extractor happens to parse correctly is exactly the
    kind of fragile-fixture trap this avoids -- this is the real code path, not an approximation
    of it. Verified against `make_six_option_item`: the stub picks Anna (A, the gold answer,
    since her profile is first to mention "born") and rejects Bruno (B) for lacking
    date_of_birth -- the rival this whole fixture is built around.
    """
    from harness import prompts
    from harness.models import StubBackend
    from harness.run_experiment import _stage1_row
    from harness.run_pilot import _StubReplies

    stub = StubBackend(_StubReplies(), name="teststub")
    response = stub.generate([prompts.render(item, "elicited")])[0]
    return _stage1_row("teststub", "stub", item, response)


def test_run_stage2_variant_records_strategy_letter_and_candidate_counts(tmp_path):
    item, rejection = make_six_option_item()
    index = build_corpus_index([item])
    row = _stage1_row_via_stub(item)
    assert row["selected"] and row["choice"] == "A" and row["selected_rival_letter"] == "B"
    stage1_rows = {item.item_id: row}

    kept, counts = run_stage2_variant(
        [item], stage1_rows, index, tmp_path, "teststub", strategy="prefer_lacking", seed=20260820,
    )
    assert counts["strategy"] == "prefer_lacking"
    assert counts["built"] == 1
    assert item.item_id in kept

    path = tmp_path / "gate8_prefer_lacking_stage2_teststub.jsonl"
    assert path.exists()
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    r = rows[0]
    assert r["target_strategy"] == "prefer_lacking"
    assert r["built"] is True
    # choice="A" equals the gold letter, so excluded={rival B, choice/gold A} removes only two
    # distinct letters, leaving all four of C/D/E/F as candidates (verified directly against
    # build_conditions_with_diagnostics above, not just asserted here).
    assert r["r3_candidates_total"] == 4
    assert r["r3_candidates_without_attribute"] == 2
    assert r["r3_picked_letter"] == "D"
    assert r["r3_strategy_recorded"] == "prefer_lacking"


def test_run_stage2_variant_never_overwrites_across_strategies(tmp_path):
    """Two strategies replayed against the same out_dir must write two distinct files, never
    one clobbering the other."""
    item, rejection = make_six_option_item()
    index = build_corpus_index([item])
    row = _stage1_row_via_stub(item)
    stage1_rows = {item.item_id: row}

    run_stage2_variant([item], stage1_rows, index, tmp_path, "teststub",
                        strategy="prefer_lacking", seed=1)
    run_stage2_variant([item], stage1_rows, index, tmp_path, "teststub",
                        strategy="random", seed=1)

    assert (tmp_path / "gate8_prefer_lacking_stage2_teststub.jsonl").exists()
    assert (tmp_path / "gate8_random_stage2_teststub.jsonl").exists()


# ------------------------------------------------------------------------------- budget / cost


def test_estimate_cost_s_is_linear_in_item_count():
    assert estimate_cost_s(0) == 0
    assert estimate_cost_s(10) == pytest.approx(10 * estimate_cost_s(1))


def test_check_budget_refuses_when_estimate_exceeds_remaining(monkeypatch):
    monkeypatch.setattr(
        "harness.gate8_variant.cumulative_gpu_seconds", lambda: C.PROJECT_GPU_BUDGET_S - 1
    )
    with pytest.raises(SystemExit):
        check_budget(estimated_s=100.0, label="test")


def test_check_budget_allows_when_estimate_fits(monkeypatch):
    monkeypatch.setattr("harness.gate8_variant.cumulative_gpu_seconds", lambda: 0.0)
    check_budget(estimated_s=100.0, label="test")  # must not raise


# --------------------------------------------------------------------------------- --part


def test_parse_part_accepts_i_of_n():
    assert _parse_part("2/3") == (1, 3)


def test_parse_part_rejects_out_of_range():
    with pytest.raises(SystemExit):
        _parse_part("0/3")
    with pytest.raises(SystemExit):
        _parse_part("4/3")


def test_parse_part_none_when_not_given():
    assert _parse_part(None) is None


def test_select_part_splits_deterministically():
    ids = ["c", "a", "b", "d", "e", "f"]  # sorted: a,b,c,d,e,f
    part1 = _select_part(ids, (0, 3))  # a, d
    part2 = _select_part(ids, (1, 3))  # b, e
    part3 = _select_part(ids, (2, 3))  # c, f
    assert part1 == ["a", "d"]
    assert part2 == ["b", "e"]
    assert part3 == ["c", "f"]
    assert sorted(part1 + part2 + part3) == sorted(ids)


def test_select_part_none_returns_everything():
    ids = ["a", "b"]
    assert _select_part(ids, None) == ids


# ---------------------------------------------------------------------------------- --dry-run


@pytest.mark.parametrize("strategy", ["prefer_lacking", "prefer_having", "random"])
def test_dry_run_end_to_end_for_every_strategy(tmp_path, strategy):
    rc = main(["--dry-run", "--strategy", strategy, "--limit", "4",
               "--out-dir", str(tmp_path / strategy)])
    assert rc == 0
    summary_path = tmp_path / strategy / f"gate8_variant_{strategy}_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["strategy"] == strategy
    assert summary["dry_run"] is True
    assert "stub" in summary["per_model"]


def test_dry_run_requires_a_strategy(tmp_path):
    with pytest.raises(SystemExit):
        main(["--dry-run", "--limit", "4", "--out-dir", str(tmp_path)])


def test_non_dry_run_requires_from_stage1(tmp_path):
    with pytest.raises(SystemExit):
        main(["--strategy", "random", "--out-dir", str(tmp_path)])
