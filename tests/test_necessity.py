"""Tests for the necessity experiment (E3): N0/N1/N2 construction, the seven integrity gates,
committed-record selection, and the end-to-end self-check/dry-run.

Hand-written fixtures only; no GPU or network.
"""

from __future__ import annotations

import json

import pytest

from pilot import config as C
from pilot.data import Entity, Item, build_item
from pilot.extract import Rejection, attribute_in_profile
from pilot.run_necessity import (
    NecessityUnavailable,
    SelectedItem,
    _select_n2_index,
    _token_len,
    build_necessity_conditions,
    check_integrity_necessity,
    find_chosen_attribute_sentence,
    is_question_relevant_attribute,
    main,
    run_stage_necessity,
    select_items,
    self_check,
)


def make_item() -> Item:
    """Chosen = A (Anna Kowalska): her own profile states date_of_birth, the attribute the
    model's rejection of the rival (B, Bruno Kowalski) names. Mirrors `test_repair.make_item`'s
    fixture shape so the two experiments' fixtures read the same way."""
    return Item(
        item_id="t1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity(
                "Anna Kowalska",
                [
                    "Anna Kowalska is a Polish filmmaker.",
                    "She was born in 1970 in Krakow.",
                    "She directed Blue River.",
                ],
            ),
            Entity(
                "Bruno Kowalski",
                [
                    "Bruno Kowalski is a Polish cinematographer.",
                    "He worked on several feature films.",
                ],
            ),
            Entity(
                "Clara Novak",
                [
                    "Clara Novak is a Czech screenwriter.",
                    "Her mother was named Maria Elena Novak.",
                ],
            ),
            Entity(
                "Dawid Kowalski",
                [
                    "Dawid Kowalski is a Polish film editor.",
                    "He edited three documentaries.",
                ],
            ),
        ],
    )


CHOSEN_LETTER = "A"
ATTRIBUTE = "date_of_birth"


# ------------------------------------------------------------------------ find_chosen_attribute


def test_find_chosen_attribute_sentence_locates_the_dated_sentence():
    item = make_item()
    found = find_chosen_attribute_sentence(item.options[0], "date_of_birth")
    assert found == (1, "She was born in 1970 in Krakow.")


def test_find_chosen_attribute_sentence_needs_a_date_for_a_dated_attribute():
    entity = Entity("Zofia Nowak", ["Zofia Nowak was born in Krakow.", "She writes novels."])
    assert find_chosen_attribute_sentence(entity, "date_of_birth") is None


def test_find_chosen_attribute_sentence_is_none_when_absent():
    item = make_item()
    assert find_chosen_attribute_sentence(item.options[0], "mother") is None


# --------------------------------------------------------------------------- build_conditions


def test_build_necessity_conditions_produces_three_clean_conditions():
    item = make_item()
    conditions, diag = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    assert set(conditions) == {"N0", "N1", "N2"}
    assert check_integrity_necessity(conditions, item, 0, ATTRIBUTE) == []
    assert diag["n1_sentence"] == "She was born in 1970 in Krakow."


def test_n1_removes_the_attribute_and_n2_does_not():
    item = make_item()
    conditions, _diag = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    titles = [o.title for o in item.options]

    assert attribute_in_profile(item.options[0].profile, ATTRIBUTE, titles) is True
    assert attribute_in_profile(conditions["N1"].options[0].profile, ATTRIBUTE, titles) is False
    assert attribute_in_profile(conditions["N2"].options[0].profile, ATTRIBUTE, titles) is True


def test_n0_is_byte_identical_to_the_unedited_item():
    item = make_item()
    conditions, _diag = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    assert [o.sentences for o in conditions["N0"].options] == [
        o.sentences for o in item.options
    ]


def test_non_chosen_profiles_are_byte_identical_across_all_three_conditions():
    item = make_item()
    conditions, _diag = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    for j in (1, 2, 3):  # every option but the chosen one
        for label in ("N0", "N1", "N2"):
            assert conditions[label].options[j].sentences == item.options[j].sentences, (j, label)


def test_option_order_titles_and_question_identical_across_conditions():
    item = make_item()
    conditions, _diag = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    titles = [o.title for o in item.options]
    for label in ("N0", "N1", "N2"):
        assert [o.title for o in conditions[label].options] == titles
        assert conditions[label].question == item.question


def test_determinism_same_input_gives_byte_identical_conditions():
    item = make_item()
    first, diag1 = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    second, diag2 = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    for label in ("N0", "N1", "N2"):
        assert [o.sentences for o in first[label].options] == [
            o.sentences for o in second[label].options
        ]
    assert diag1 == diag2


def test_drops_when_the_chosen_profile_lacks_the_named_attribute():
    item = make_item()
    with pytest.raises(NecessityUnavailable) as exc_info:
        build_necessity_conditions(item, CHOSEN_LETTER, "mother")  # Anna never mentions a mother
    assert "not stated in one isolable sentence" in str(exc_info.value)


def test_drops_when_the_chosen_profile_has_fewer_than_two_sentences():
    item = Item(
        item_id="t2", question="q", answer="Anna Kowalska", gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["She was born in 1970 in Krakow."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
        ],
    )
    with pytest.raises(NecessityUnavailable) as exc_info:
        build_necessity_conditions(item, "A", "date_of_birth")
    assert "fewer than two sentences" in str(exc_info.value)


# ----------------------------------------------------------------------- the seven gates


def test_gate1_fails_when_the_attribute_is_still_present_after_n1():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N1"].options[0].sentences.insert(1, "She was born in 1970 in Krakow.")
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate1:") for f in failures)


def test_gate2_fails_when_n2_also_removes_the_named_attribute():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N2"].options[0].sentences = [
        "Anna Kowalska is a Polish filmmaker.", "She directed Blue River.",
    ]
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate2:") for f in failures)


def test_gate3_fails_when_n1_n2_lengths_diverge_by_more_than_20_percent():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N2"].options[0].sentences = [
        "Anna Kowalska is a Polish filmmaker.", "She was born in 1970 in Krakow.",
    ]  # removes "She directed Blue River." (4 tokens) instead of the 7-token N1 sentence
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate3:") for f in failures)


def test_gate4_fails_when_a_condition_leaves_no_sentences():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N1"].options[0].sentences = []
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate4:") for f in failures)


def test_gate5_fails_when_the_deleted_sentence_names_another_candidate():
    item = make_item()
    item.options[0].sentences.insert(2, "She once worked alongside Bruno Kowalski on a project.")
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N2"] = Item(
        item_id=item.item_id, question=item.question, answer=item.answer,
        gold_title=item.gold_title,
        options=[
            Entity(o.title, list(o.sentences)) for o in item.options
        ],
    )
    conditions["N2"].options[0].sentences = [
        s for s in item.options[0].sentences if "Bruno Kowalski" not in s
    ]
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate5:") for f in failures)


def test_gate6_fails_when_a_non_chosen_profile_is_touched():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N1"].options[1].sentences.append("This should never be here.")
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate6:") for f in failures)


def test_gate7_fails_when_option_order_differs_across_conditions():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N1"].options[0], conditions["N1"].options[1] = (
        conditions["N1"].options[1], conditions["N1"].options[0],
    )
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate7:") for f in failures)


def test_gate7_fails_when_the_question_differs_across_conditions():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    conditions["N1"].question = "A different question entirely?"
    failures = check_integrity_necessity(conditions, item, 0, ATTRIBUTE)
    assert any(f.startswith("gate7:") for f in failures)


def test_all_seven_gates_pass_on_a_clean_build():
    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    assert check_integrity_necessity(conditions, item, 0, ATTRIBUTE) == []


# ------------------------------------------------------------------------- 20% boundary


def _words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_gate3_boundary_exactly_20_percent_passes():
    # N1's sentence is exactly 10 tokens ("Her mother " + 8 filler words); tolerance is 2.0
    # tokens. A sole candidate at exactly 12 tokens (diff == 2, the boundary) must still build.
    # Both end in a period: `Entity.profile` joins `sentences` with spaces and `split_sentences`
    # re-splits on sentence-ending punctuation, so two entries with none of that would be read
    # back as a single merged sentence rather than the two isolable entries this test needs.
    n1_sentence = "Her mother " + _words("w", 8) + "."
    assert _token_len(n1_sentence) == 10
    candidate = _words("x", 12) + "."
    assert _token_len(candidate) == 12

    item = Item(
        item_id="boundary-ok", question="q", answer="Anna", gold_title="Anna",
        options=[
            Entity("Anna", [n1_sentence, candidate]),
            Entity("Bruno", ["Bruno has no attribute stated here at all."]),
        ],
    )
    conditions, diag = build_necessity_conditions(item, "A", "mother")
    assert diag["n2_sentence"] == candidate
    assert check_integrity_necessity(conditions, item, 0, "mother") == []


def test_gate3_boundary_one_token_more_fails():
    # Same N1 sentence (10 tokens, tolerance 2.0); the only candidate is 13 tokens (diff == 3,
    # one more than the boundary) -- must be refused, not silently accepted.
    n1_sentence = "Her mother " + _words("w", 8) + "."
    assert _token_len(n1_sentence) == 10
    candidate = _words("y", 13) + "."
    assert _token_len(candidate) == 13

    item = Item(
        item_id="boundary-fail", question="q", answer="Anna", gold_title="Anna",
        options=[
            Entity("Anna", [n1_sentence, candidate]),
            Entity("Bruno", ["Bruno has no attribute stated here at all."]),
        ],
    )
    with pytest.raises(NecessityUnavailable) as exc_info:
        build_necessity_conditions(item, "A", "mother")
    exc = exc_info.value
    assert exc.gate_failures is not None
    assert any(f.startswith("gate3:") for f in exc.gate_failures)


# -------------------------------------------------------------------------- N2 selection


def test_select_n2_index_prefers_the_closest_length_match():
    item = make_item()
    entity = item.options[0]
    n1_index, n1_sentence = find_chosen_attribute_sentence(entity, "date_of_birth")
    target_len = _token_len(n1_sentence)
    idx = _select_n2_index(entity, n1_index, "date_of_birth", target_len)
    assert idx == 0  # "Anna Kowalska is a Polish filmmaker." (6 tokens) beats "She directed
    # Blue River." (4 tokens) against a 7-token target


# ------------------------------------------------------------------ question-relevance flag


@pytest.mark.parametrize("question,attribute,expected", [
    ("Who was born first, Anna or Bruno?", "date_of_birth", True),
    ("Which of them died earlier?", "date_of_death", True),
    ("Who directed the film Blue River?", "date_of_birth", False),  # no order cue
    ("Who was born first, Anna or Bruno?", "mother", False),  # not a dated attribute
])
def test_is_question_relevant_attribute(question, attribute, expected):
    assert is_question_relevant_attribute(question, attribute) is expected


# --------------------------------------------------------------- select_items (committed records)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _det_record() -> dict:
    return {
        "_id": "detitem-1",
        "question": "Who directed the film Cedar Hollow?",
        "answer": "Anna Nowak",
        "context": [
            ["Anna Nowak", ["Anna Nowak is a Polish director.", "She directed Cedar Hollow."]],
            ["Bruno Zielinski", ["Bruno Zielinski is a Polish actor.",
                                  "He appeared in several plays."]],
            ["Karolina Wozniak", ["Karolina Wozniak is a costume designer.",
                                   "Her mother was a seamstress in Lodz."]],
            ["Dawid Krol", ["Dawid Krol is a sound engineer.", "He recorded several albums."]],
        ],
    }


def _make_committed_run(tmp_path):
    """A synthetic exp3a directory: one data file, one stage1 row for one model, built so E3's
    population conditions all hold -- chosen (Karolina) is not the gold answer (Anna), and her
    own profile states the attribute ("mother") the response names as missing from the rejected
    rival (Bruno)."""
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(json.dumps(_det_record()) + "\n", encoding="utf-8")

    item = build_item(_det_record(), n_options=4, seed=C.SEED)
    chosen_letter = item.letter_of("Karolina Wozniak")
    assert chosen_letter != item.gold_letter  # the case this fixture is built to exercise

    response = (
        f"Answer: {chosen_letter}) Karolina Wozniak.\n"
        "Karolina Wozniak fits the question well. It is not Bruno Zielinski, because that "
        "profile never mentions his mother."
    )
    row = {
        "model": "test-model", "item_id": item.item_id, "choice": chosen_letter,
        "option_titles": [o.title for o in item.options], "selected": True, "response": response,
    }
    exp_dir = tmp_path / "results" / "exp3a"
    _write_jsonl(exp_dir / "stage1_test-model.jsonl", [row])
    return tmp_path / "results", data_file, item, chosen_letter


def test_select_items_reads_committed_stage1_records():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results_dir, data_file, item, chosen_letter = _make_committed_run(tmp_path)

        selected, counts = select_items(
            part="A", results_dir=results_dir, data_file=data_file, models=["test-model"],
        )
        assert "test-model" in selected
        kept = selected["test-model"]
        assert item.item_id in kept
        sel = kept[item.item_id]
        assert isinstance(sel, SelectedItem)
        assert sel.chosen_letter == chosen_letter
        assert sel.chosen_is_gold is False  # Karolina is not the gold answer, Anna is
        assert sel.rejection.attribute == "mother"
        assert sel.question_relevant_attribute is False  # "mother" is not a dated attribute
        assert counts["models"]["test-model"]["kept"] == 1


def test_select_items_is_deterministic_given_the_seed():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results_dir, data_file, item, _chosen_letter = _make_committed_run(tmp_path)

        first, _ = select_items(
            part="A", results_dir=results_dir, data_file=data_file, models=["test-model"],
        )
        second, _ = select_items(
            part="A", results_dir=results_dir, data_file=data_file, models=["test-model"],
        )
        first_sel = first["test-model"][item.item_id]
        second_sel = second["test-model"][item.item_id]
        assert first_sel.chosen_letter == second_sel.chosen_letter
        assert first_sel.chosen_is_gold == second_sel.chosen_is_gold
        for label in ("N0", "N1", "N2"):
            assert [o.sentences for o in first_sel.conditions[label].options] == [
                o.sentences for o in second_sel.conditions[label].options
            ]


# ---------------------------------------------------------------------------- run_stage_necessity


def test_run_stage_necessity_records_still_chooses_original_per_condition(tmp_path):
    from pilot.models import StubBackend

    item = make_item()
    conditions, _ = build_necessity_conditions(item, CHOSEN_LETTER, ATTRIBUTE)
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B", title="Bruno Kowalski", matched_by="title", attribute=ATTRIBUTE,
        cue="birth", negation_cue="not",
    )
    sel = SelectedItem(
        item=item, rejection=rejection, chosen_letter="A", conditions=conditions,
        chosen_is_gold=True, question_relevant_attribute=False,
    )
    kept = {item.item_id: sel}

    scripted = ["A", "B", "A"]  # N0, N1, N2

    class _Scripted:
        def __init__(self, letters):
            self.letters = list(letters)

        def __call__(self, _index, _chat):
            return f"The answer is {self.letters.pop(0)}."

    backend = StubBackend(_Scripted(scripted), name="scripted")
    outcomes, counts = run_stage_necessity(backend, kept, tmp_path, part=None)

    assert counts["n_items"] == 1
    assert outcomes[item.item_id] == {"N0": True, "N1": False, "N2": True}
    assert counts["n0_reproduced"] == 1
    assert counts["n0_flipped"] == 0

    rows = [
        json.loads(l)
        for l in (tmp_path / "stage_necessity_scripted.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    by_condition = {r["condition"]: r for r in rows}
    assert set(by_condition) == {"N0", "N1", "N2"}
    assert by_condition["N1"]["still_chooses_original"] is False
    assert by_condition["N1"]["chosen_letter"] == "A"
    assert by_condition["N0"]["delta_p_original"] == 0.0


# --------------------------------------------------------------------------------- self-check


def test_self_check_passes():
    assert self_check() == 0


# ------------------------------------------------------------------------------- --dry-run


def test_dry_run_end_to_end_produces_a_summary(tmp_path):
    rc = main(["--dry-run", "--out-dir", str(tmp_path)])
    assert rc == 0

    summary = json.loads((tmp_path / "necessity_summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["seed"] == C.SEED
    assert "stub" in summary["per_model"]
    assert "condition_counts" in summary["per_model"]["stub"]
    assert "mcnemar_n1_vs_n2" in summary["per_model"]["stub"]


def test_dry_run_respects_limit(tmp_path):
    rc = main(["--dry-run", "--limit", "1", "--out-dir", str(tmp_path)])
    assert rc == 0
    summary = json.loads((tmp_path / "necessity_summary.json").read_text(encoding="utf-8"))
    assert summary["n_items"] == 1
