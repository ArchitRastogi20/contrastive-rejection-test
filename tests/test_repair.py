"""Tests for harness.repair: builds conditions R0-R4 and checks the eight integrity gates.

Hand-written fixtures only; no GPU or network."""

import pytest

from harness.data import Entity, Item
from harness.extract import Rejection, attribute_in_profile
from harness.repair import (
    R1_SEARCH_CAP,
    R2_SEARCH_CAP,
    RepairUnavailable,
    build_conditions,
    build_conditions_with_diagnostics,
    build_corpus_index,
    check_integrity,
    classify_stratum,
    find_attribute_sentence,
    retarget,
    select_repair_source,
    _token_len,
)


def make_item() -> Item:
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
        relations=[
            ("Blue River", "director", "Anna Kowalska"),
            ("Anna Kowalska", "date of birth", "1970"),
        ],
    )


def make_rejection() -> Rejection:
    return Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="not",
    )


def make_conditions(item: Item, rejection: Rejection) -> dict:
    index = build_corpus_index([item])
    return build_conditions(item, rejection, index, choice_letter="A")


def make_true_value_item(with_evidence: bool) -> tuple[Item, Rejection]:
    """Bruno is missing a date of death. Two siblings each carry one: Clara's is not his real
    value, Dawid's (1985) is, per the evidence triple -- present only when `with_evidence`.
    Elena carries an unrelated attribute so R2 has somewhere to source from. Grazyna carries
    no attribute at all and sits right after the rival so she -- not Clara, who is a candidate
    R1 source and already carries the named attribute -- becomes R3/R4's target under the
    exclusion rule (excluded={gold, rival}); R4's gate would otherwise reject the item outright
    since Clara's own profile already has a date of death before any edit."""
    item = Item(
        item_id="tv1",
        question="When did Bruno Kowalski die?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity(
                "Anna Kowalska",
                ["Anna Kowalska is a Polish filmmaker.", "She directed Blue River."],
            ),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            Entity("Grazyna Wojcik", ["Grazyna Wojcik lives quietly in a small town."]),
            Entity("Clara Novak", ["Clara Novak died in 1950 in Prague."]),
            Entity("Dawid Kowalski", ["Dawid Kowalski died in 1985 in Warsaw."]),
            Entity("Elena Nowak", ["Elena Nowak worked as a film editor."]),
        ],
        relations=[("Bruno Kowalski", "date of death", "1985")] if with_evidence else [],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of death.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_death",
        cue="death",
        negation_cue="never",
    )
    return item, rejection


# --------------------------------------------------------------------------------- retarget


def test_retarget_replaces_a_named_subject():
    out = retarget(
        "Bruno Kowalski is a Polish cinematographer.", "Bruno Kowalski", "George Whitaker"
    )
    assert out == "George Whitaker is a Polish cinematographer."


def test_retarget_replaces_a_leading_surname_only_mention():
    out = retarget("Kowalski directed several films.", "Bruno Kowalski", "George Whitaker")
    assert out == "George Whitaker directed several films."


def test_retarget_replaces_a_leading_pronoun_and_fixes_the_copula():
    assert (
        retarget("He was born in 1980.", "X Y", "George Whitaker")
        == "George Whitaker was born in 1980."
    )
    assert (
        retarget("They were the founders.", "X Y", "George Whitaker")
        == "George Whitaker was the founders."
    )


def test_retarget_does_not_grab_a_name_that_is_not_the_subject():
    # "Novak" appears in the sentence but names the mother, not the subject -- the subject is
    # the leading pronoun "Her". A token match anywhere in the sentence would corrupt this,
    # which is exactly what happened before the leading-position restriction was added.
    out = retarget("Her mother was named Maria Elena Novak.", "Clara Novak", "George Whitaker")
    assert out == "George Whitaker mother was named Maria Elena Novak."


def test_retarget_returns_none_when_neither_mechanical_case_applies():
    assert retarget("The weather was nice that day.", "Bruno Kowalski", "George Whitaker") is None


# ------------------------------------------------------------------ find_attribute_sentence


def test_find_attribute_sentence_needs_a_date_for_a_dated_attribute():
    # "born in Krakow" does not answer when someone was born -- same rule as attribute_in_profile.
    entity = Entity("Zofia Nowak", ["Zofia Nowak was born in Krakow.", "She writes novels."])
    assert find_attribute_sentence(entity, "date_of_birth") is None


def test_find_attribute_sentence_finds_the_dated_sentence():
    entity = Entity(
        "Zofia Nowak", ["Zofia Nowak was born in 1955 in Krakow.", "She writes novels."]
    )
    assert (
        find_attribute_sentence(entity, "date_of_birth")
        == "Zofia Nowak was born in 1955 in Krakow."
    )


def test_find_attribute_sentence_is_none_when_the_attribute_is_absent():
    entity = Entity("Zofia Nowak", ["Zofia Nowak writes novels."])
    assert find_attribute_sentence(entity, "mother") is None


# ------------------------------------------------------------------------- build_conditions


def test_build_conditions_produces_four_clean_conditions():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    assert set(conditions) == {"R0", "R1", "R2", "R3", "R4"}
    assert check_integrity(conditions, item, rejection) == []


def test_r2_does_not_accidentally_repair_the_named_attribute():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    titles = [o.title for o in item.options]
    rival_in_r2 = conditions["R2"].options[1]
    assert attribute_in_profile(rival_in_r2.profile, "date_of_birth", titles) is False


def test_r1_and_r2_inserted_sentences_are_length_matched_within_tolerance():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    len1 = _token_len(conditions["R1"].options[1].sentences[-1])
    len2 = _token_len(conditions["R2"].options[1].sentences[-1])
    assert abs(len1 - len2) <= 0.20 * len1 + 1e-9


def test_r3_edits_a_different_option_and_leaves_the_rival_untouched():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    r3 = conditions["R3"]
    assert r3.options[1].sentences == item.options[1].sentences  # rival: untouched
    touched = [i for i in range(4) if r3.options[i].sentences != item.options[i].sentences]
    assert touched == [2]  # Clara Novak: the only non-chosen, non-gold, non-rival option


def test_determinism_same_input_gives_byte_identical_conditions():
    item = make_item()
    rejection = make_rejection()
    index = build_corpus_index([item])
    first = build_conditions(item, rejection, index, choice_letter="A")
    second = build_conditions(item, rejection, index, choice_letter="A")
    for label in ("R0", "R1", "R2", "R3", "R4"):
        assert [o.sentences for o in first[label].options] == [
            o.sentences for o in second[label].options
        ]


# ----------------------------------------------------------------------- the eight gates


def test_gate1_fails_when_the_attribute_was_never_actually_added():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R1"].options[1].sentences.pop()  # undo the repair
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate1:") for f in failures)


def test_gate2_fails_when_r2_accidentally_repairs_the_named_attribute():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R2"].options[1].sentences.append("Bruno Kowalski was born in 1970 in Krakow.")
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate2:") for f in failures)


def test_gate3_fails_when_r1_r2_lengths_diverge_by_more_than_20_percent():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R2"].options[1].sentences[-1] += " " + "word " * 40
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate3:") for f in failures)


def test_gate4_fails_on_an_internally_contradictory_date():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R1"].options[1].sentences[-1] = "Bruno Kowalski was born in 1999."
    conditions["R1"].options[1].sentences.append("Bruno Kowalski died in 1950.")
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate4:") for f in failures)


def test_gate5_fails_when_the_inserted_sentence_names_another_candidate():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R1"].options[1].sentences[-1] = (
        "Bruno Kowalski was born the same year as Clara Novak."
    )
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate5:") for f in failures)


def test_gate6_fails_when_the_gold_profile_is_touched():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R1"].options[0].sentences.append("This sentence should never be here.")
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate6:") for f in failures)


def test_gate7_fails_when_option_order_differs_across_conditions():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R1"].options[0], conditions["R1"].options[1] = (
        conditions["R1"].options[1],
        conditions["R1"].options[0],
    )
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate7:") for f in failures)


def test_gate8_fails_when_r4_accidentally_repairs_the_named_attribute():
    # R4's target here is Clara Novak (C): excluded={A, B} under choice_letter="A". Appending a
    # sentence that carries the named attribute (date_of_birth) to her profile is R4's
    # counterpart of gate 2's "R2 must not accidentally repair the rival" check.
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R4"].options[2].sentences.append("Clara Novak was born in 1970 in Krakow.")
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate8:") for f in failures)


def test_gate8_fails_when_r4_edits_a_different_option_than_r3():
    item = make_item()
    rejection = make_rejection()
    conditions = make_conditions(item, rejection)
    conditions["R4"].options[2].sentences.pop()  # undo R4's real edit on Clara (R3's target)
    conditions["R4"].options[3].sentences.append("An unrelated aside about Dawid.")  # edit D instead
    failures = check_integrity(conditions, item, rejection)
    assert any(f.startswith("gate8:") for f in failures)


# --------------------------------------------------------------------- the true-value stratum


def test_true_value_stratum_sources_the_sentence_carrying_the_real_value():
    item, rejection = make_true_value_item(with_evidence=True)
    index = build_corpus_index([item])

    picked = select_repair_source(item, rejection, index)
    assert picked is not None
    source_entity, sentence, stratum = picked
    assert stratum == "true_value"
    assert source_entity.title == "Dawid Kowalski"
    assert "1985" in sentence

    conditions = build_conditions(item, rejection, index, choice_letter="A")
    assert "1985" in conditions["R1"].options[1].sentences[-1]
    assert classify_stratum(item, rejection, index) == "true_value"


def test_borrowed_stratum_still_builds_when_no_real_value_is_known():
    item, rejection = make_true_value_item(with_evidence=False)
    index = build_corpus_index([item])

    picked = select_repair_source(item, rejection, index)
    assert picked is not None
    source_entity, sentence, stratum = picked
    assert stratum == "borrowed"
    assert source_entity.title == "Clara Novak"  # first candidate in option order

    conditions = build_conditions(item, rejection, index, choice_letter="A")
    assert "1950" in conditions["R1"].options[1].sentences[-1]
    assert classify_stratum(item, rejection, index) == "borrowed"


# ------------------------------------------------------- the joint R1 x R2 search advances on

# These fixtures each stack a sibling that carries the named attribute but cannot be
# mechanically retargeted (no self-mention, no leading pronoun) ahead of one that can, in
# option order -- the case the defect this fix targets used to drop the whole item on.


def test_r1_search_advances_past_a_non_retargetable_first_candidate():
    item = Item(
        item_id="r1adv",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            # No attribute at all, so under excluded={gold, rival} she -- not Elena, who
            # already carries the named attribute -- becomes R3/R4's target.
            Entity("Grazyna Wojcik", ["Grazyna Wojcik lives quietly in a small town."]),
            Entity(
                "Elena Sikorska",
                ["A record from that era gives the year 1930 as a birth date."],
            ),
            Entity(
                "Feliks Nowak",
                ["Feliks Nowak was born in 1931.", "He worked as a translator."],
            ),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    index = build_corpus_index([item])

    # the naive head pick (what the old single-shot code would have used, and dropped on)
    # is Elena's sentence, and it is genuinely not retargetable
    head_entity, head_sentence, _stratum = select_repair_source(item, rejection, index)
    assert head_entity.title == "Elena Sikorska"
    assert retarget(head_sentence, head_entity.title, "Bruno Kowalski") is None

    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    assert diag.r1_examined == 2  # Elena tried and failed, Feliks tried and succeeded
    assert conditions["R1"].options[1].sentences[-1] == "Bruno Kowalski was born in 1931."
    assert check_integrity(conditions, item, rejection) == []


def test_r2_search_advances_past_a_non_retargetable_first_candidate():
    item = Item(
        item_id="r2adv",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity(
                "Anna Kowalska",
                ["Anna Kowalska is a Polish filmmaker who works mostly in Warsaw."],
            ),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            # not retargetable: no self-mention, no leading pronoun -- but it is the closer
            # length match to R1's sentence, so the old single-shot code would have picked it
            Entity("Grazyna Wojcik", ["Local gossip said someone married quickly."]),
            # retargetable, but a slightly worse length match, so only reached by advancing
            Entity("Henryk Baran", ["Henryk Baran once worked as a translator."]),
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    index = build_corpus_index([item])

    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    assert diag.r1_examined == 1
    assert diag.r2_examined == 2  # Grazyna tried and failed, Henryk tried and succeeded
    assert conditions["R1"].options[1].sentences[-1] == "Bruno Kowalski was born in 1931."
    assert (
        conditions["R2"].options[1].sentences[-1]
        == "Bruno Kowalski once worked as a translator."
    )
    assert check_integrity(conditions, item, rejection) == []


def test_r2_search_advances_past_a_candidate_that_would_trip_gate2():
    """R2's first-ranked candidate happens to carry a bare year, which -- combined with the
    rival's own pre-existing, date-less "born" sentence -- would satisfy `attribute_in_profile`
    for the named attribute and trip gate 2. A later candidate with no date in it does not, and
    the search reaches it instead of dropping the item."""
    item = Item(
        item_id="gate2adv",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity(
                "Bruno Kowalski",
                [
                    "Bruno Kowalski is a Polish cinematographer.",
                    "Bruno Kowalski was born in Krakow.",  # the cue "born", but no date: a
                ],  # true complaint (attribute_in_profile is False) until an R2 edit adds a date
            ),
            Entity("Henryk Baran", ["He worked as guy in 1950."]),  # ranked first, has a year
            Entity("Grazyna Wojcik", ["Grazyna Wojcik worked as a librarian nicely."]),
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    index = build_corpus_index([item])

    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    assert diag.r2_examined == 2  # Henryk tried (gate2) and failed, Grazyna tried and succeeded
    assert conditions["R2"].options[1].sentences[-1] == "Bruno Kowalski worked as a librarian nicely."
    assert check_integrity(conditions, item, rejection) == []


def test_gate2_still_fires_and_drops_the_item_when_no_candidate_avoids_it():
    """Same fixture as above, minus the safe candidate: every available R2 candidate
    accidentally repairs the named attribute, so the item is genuinely exhausted and drops --
    gate 2 was not weakened to make something build."""
    item = Item(
        item_id="gate2fail",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity(
                "Bruno Kowalski",
                ["Bruno Kowalski is a Polish cinematographer.",
                 "Bruno Kowalski was born in Krakow."],
            ),
            Entity("Henryk Baran", ["He worked as guy in 1950."]),
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable) as exc_info:
        build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    exc = exc_info.value
    assert exc.gate_failures is not None
    assert any(f.startswith("gate2:") for f in exc.gate_failures)
    assert exc.cap_hit is False


def test_exhaustion_drops_the_item_with_no_alternate_attribute_source():
    """No sibling anywhere in the item carries any attribute other than the named one, so R2
    has nowhere to source from -- a different exhaustion reason than gate 2's, still reported
    precisely rather than as a generic failure."""
    item = Item(
        item_id="nor2src",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable) as exc_info:
        build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    assert "no alternate-attribute source available for R2" in str(exc_info.value)


def test_determinism_holds_when_the_search_has_to_advance():
    """The determinism guarantee is not just for the trivial, first-candidate-works path --
    building twice through a search that has to reject a candidate and advance must still give
    byte-identical conditions."""
    item = Item(
        item_id="r1adv-det",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            # No attribute at all, so under excluded={gold, rival} she -- not Elena, who
            # already carries the named attribute -- becomes R3/R4's target.
            Entity("Grazyna Wojcik", ["Grazyna Wojcik lives quietly in a small town."]),
            Entity(
                "Elena Sikorska",
                ["A record from that era gives the year 1930 as a birth date."],
            ),
            Entity(
                "Feliks Nowak",
                ["Feliks Nowak was born in 1931.", "He worked as a translator."],
            ),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    index = build_corpus_index([item])
    first, diag1 = build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    second, diag2 = build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    for label in ("R0", "R1", "R2", "R3", "R4"):
        assert [o.sentences for o in first[label].options] == [
            o.sentences for o in second[label].options
        ]
    assert diag1 == diag2


# ------------------------------------------------------------------------- the search cap


def _archival_entries(n: int) -> list[Entity]:
    # 6 tokens, no self-mention, no leading pronoun: carries "occupation" but cannot retarget.
    return [Entity(f"Archival Entry {i}", ["Nobody local worked as a clerk."]) for i in range(n)]


def test_search_cap_is_honoured_not_just_declared():
    """40 non-retargetable R2 candidates rank ahead of 5 that *would* work: if the search kept
    going past `R2_SEARCH_CAP` it would find one of them and build. It must not -- the item is
    genuinely capped-out, not genuinely exhausted, but the drop is the same either way."""
    good_sentence = "He worked as a " + "very " * 15 + "long career total."
    options = [
        Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
        Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
        Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        *_archival_entries(R2_SEARCH_CAP),
        *[Entity(f"Recorded Case {i}", [good_sentence]) for i in range(5)],
    ]
    item = Item(item_id="cap1", question="q", answer="Bruno Kowalski", gold_title="Anna Kowalska",
                options=options)
    rejection = Rejection(
        sentence="x", letter="B", title="Bruno Kowalski", matched_by="title",
        attribute="date_of_birth", cue="birth", negation_cue="never",
    )
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable) as exc_info:
        build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    exc = exc_info.value
    assert "R2 sentence is not mechanically" in str(exc)
    assert exc.cap_hit is True


def test_search_cap_hit_is_recorded_on_a_built_item():
    """The cap can be hit and the item can still build: a working candidate ranked at exactly
    `R2_SEARCH_CAP`, with one more candidate beyond it that is never reached. `cap_hit` records
    that the pool was truncated even though truncation did not end up costing this item."""
    item = Item(
        item_id="cap2",
        question="q",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            *_archival_entries(R2_SEARCH_CAP - 1),
            Entity("Special Witness", ["He worked as a witness nicely."]),  # rank R2_SEARCH_CAP
            Entity(  # rank R2_SEARCH_CAP + 1: beyond the cap, never reached
                "Distant Filler", ["He worked as a " + "very " * 15 + "long career total."]
            ),
            # R1's only source, moved to the end (rather than right after the rival) so an
            # archival entry -- not Feliks, who already carries the named attribute -- becomes
            # R3/R4's target under excluded={gold, rival}. Neither the R2 pool (which already
            # excludes the named attribute) nor the search-order assertions below depend on
            # her list position.
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        ],
    )
    rejection = Rejection(
        sentence="x", letter="B", title="Bruno Kowalski", matched_by="title",
        attribute="date_of_birth", cue="birth", negation_cue="never",
    )
    index = build_corpus_index([item])
    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    assert diag.r2_total == R2_SEARCH_CAP + 1
    assert diag.r2_examined == R2_SEARCH_CAP
    assert diag.cap_hit is True
    assert conditions["R2"].options[1].sentences[-1] == "Bruno Kowalski worked as a witness nicely."
    assert check_integrity(conditions, item, rejection) == []


# ---------------------------------------------------------------- an unreadable stage-1 choice


def test_build_conditions_drops_an_unreadable_choice_rather_than_guessing_gold():
    item = make_item()
    rejection = make_rejection()
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable):
        build_conditions(item, rejection, index, choice_letter=None)


# ---------------------------------------------------------------------------------- R4, the 2x2


def make_r4_item() -> tuple[Item, Rejection]:
    """A fixture where R1/R2's source entities differ from R3/R4's target, so the four inserted
    sentences are visibly distinct text rather than the coincidental self-retarget that happens
    in `make_item()` (there, R2's source is Clara Novak and Clara Novak is also R3's target, so
    R4 retargets her own sentence onto herself). Rival: Bruno Kowalski, missing date_of_birth.
    R3/R4's target (excluded={A, B} under choice_letter="A") is Grazyna Wojcik (C), who carries
    no attribute of her own. R1's only source is Feliks Nowak (date_of_birth); R2's only source
    is Henryk Baran (occupation)."""
    item = Item(
        item_id="r4clean",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            Entity("Grazyna Wojcik", ["Grazyna Wojcik lives quietly in a small town."]),
            Entity("Henryk Baran", ["Henryk Baran worked as a witness nicely."]),
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    return item, rejection


def test_r4_reuses_r2s_sentence_retargeted_onto_r3s_target():
    item, rejection = make_r4_item()
    index = build_corpus_index([item])
    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")

    assert check_integrity(conditions, item, rejection) == []
    # R2 and R4 both trace back to Henryk Baran's "worked as a witness nicely" -- same tail,
    # different subject -- never an independently searched sentence for R4.
    assert conditions["R2"].options[1].sentences[-1] == "Bruno Kowalski worked as a witness nicely."
    assert conditions["R4"].options[2].sentences[-1] == "Grazyna Wojcik worked as a witness nicely."
    # R1 and R3 both trace back to Feliks Nowak's "was born in 1931" -- the existing pairing
    # R4 is designed to mirror.
    assert conditions["R1"].options[1].sentences[-1] == "Bruno Kowalski was born in 1931."
    assert conditions["R3"].options[2].sentences[-1] == "Grazyna Wojcik was born in 1931."


def test_2x2_structural_relationship_holds():
    """R1/R2 share a target (the rival); R3/R4 share a target (R3's target); R1/R3 share a
    sentence (up to subject); R2/R4 share a sentence (up to subject) -- the 2x2 the experiment
    design doc and repair.py's module docstring describe."""
    item, rejection = make_r4_item()
    index = build_corpus_index([item])
    conditions = build_conditions(item, rejection, index, choice_letter="A")

    def edited_index(cond_item):
        for i, (orig, new) in enumerate(zip(item.options, cond_item.options)):
            if orig.sentences != new.sentences:
                return i
        return None

    r1_idx, r2_idx = edited_index(conditions["R1"]), edited_index(conditions["R2"])
    r3_idx, r4_idx = edited_index(conditions["R3"]), edited_index(conditions["R4"])
    assert r1_idx == r2_idx == 1  # the rival, Bruno Kowalski
    assert r3_idx == r4_idx == 2  # Grazyna Wojcik
    assert r1_idx != r3_idx  # location genuinely differs between the two pairs

    def tail_after_subject(sentence: str, subject: str) -> str:
        return sentence[len(subject):]

    r1_tail = tail_after_subject(conditions["R1"].options[r1_idx].sentences[-1], "Bruno Kowalski")
    r3_tail = tail_after_subject(conditions["R3"].options[r3_idx].sentences[-1], "Grazyna Wojcik")
    assert r1_tail == r3_tail == " was born in 1931."

    r2_tail = tail_after_subject(conditions["R2"].options[r2_idx].sentences[-1], "Bruno Kowalski")
    r4_tail = tail_after_subject(conditions["R4"].options[r4_idx].sentences[-1], "Grazyna Wojcik")
    assert r2_tail == r4_tail == " worked as a witness nicely."
    assert r1_tail != r2_tail  # content genuinely differs between the two pairs


def test_r4_gate_failure_drops_the_whole_item_with_integrity_gate_failed_reason():
    """All three non-excluded options (Grazyna, Henryk, Feliks) already carry the named
    attribute in their own base profile, before any edit -- so the new "prefer a candidate that
    lacks the attribute" selection has nothing to prefer, falls back to the first one (Grazyna,
    same as the old unconditional rule would pick), and every R2/R4 combination still trips
    gate 8 regardless of which source sentence is tried. The item drops as a whole rather than
    being kept with four conditions and a missing fifth -- unchanged from before this fix,
    because there was genuinely no better candidate to select."""
    from harness.run_experiment import _drop_category

    item = Item(
        item_id="r4gate8drop",
        question="When was Bruno Kowalski born?",
        answer="Bruno Kowalski",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            Entity("Grazyna Wojcik", ["Grazyna Wojcik was born in 1928."]),
            # carries date_of_birth (so the new selection has nothing to prefer here either)
            # and, separately, an occupation sentence so R2 still has somewhere to source from.
            Entity("Henryk Baran", ["Henryk Baran was born in 1929.",
                                     "Henryk Baran worked as a witness nicely."]),
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1931."]),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B",
        title="Bruno Kowalski",
        matched_by="title",
        attribute="date_of_birth",
        cue="birth",
        negation_cue="never",
    )
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable) as exc_info:
        build_conditions_with_diagnostics(item, rejection, index, choice_letter="A")
    exc = exc_info.value
    assert exc.gate_failures is not None
    assert any(f.startswith("gate8:") for f in exc.gate_failures)
    assert _drop_category(str(exc)) == "integrity_gate_failed"


def test_r4_determinism_same_input_gives_byte_identical_conditions():
    item, rejection = make_r4_item()
    index = build_corpus_index([item])
    first = build_conditions(item, rejection, index, choice_letter="A")
    second = build_conditions(item, rejection, index, choice_letter="A")
    for label in ("R0", "R1", "R2", "R3", "R4"):
        assert [o.sentences for o in first[label].options] == [
            o.sentences for o in second[label].options
        ]


# --------------------------------------------------- R3/R4 target selection (n_options gate 8)

# The fix under test: with more than 4 options, more than one non-excluded candidate can exist
# for R3/R4. Selection must prefer one that already lacks the named attribute (so gate 8 passes
# by construction), with a deterministic first-in-option-order tie-break, and must fall back to
# the old "first available" rule -- never inventing a third behaviour -- when no candidate
# qualifies, letting gate 8 drop the item exactly as it does today.


def make_forced_single_candidate_item(d_carries_attribute: bool) -> tuple[Item, Rejection]:
    """4 options, `choice_letter="C"` (distinct from the gold letter "A"): excluded =
    {A, B, C} leaves exactly one candidate, D -- the genuine n_options=4 forced case, as
    opposed to the fixtures above whose `choice_letter` happens to equal the gold letter and so
    leave two candidates. R1's source is the gold entity's own pronoun-led birth sentence,
    exactly as in `make_item()`; R2's source is Clara's "mother" sentence."""
    item = Item(
        item_id="forced1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity(
                "Anna Kowalska",
                ["Anna Kowalska is a Polish filmmaker.", "She was born in 1970 in Krakow.",
                 "She directed Blue River."],
            ),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            Entity("Clara Novak", ["Clara Novak is a Czech screenwriter.",
                                    "Her mother was named Maria Elena Novak."]),
            Entity(
                "Dawid Kowalski",
                ["Dawid Kowalski was born in 1928."] if d_carries_attribute
                else ["Dawid Kowalski is a Polish film editor.", "He edited three documentaries."],
            ),
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B", title="Bruno Kowalski", matched_by="title", attribute="date_of_birth",
        cue="birth", negation_cue="not",
    )
    return item, rejection


def test_n_options_4_forced_candidate_that_lacks_the_attribute_builds_as_before():
    item, rejection = make_forced_single_candidate_item(d_carries_attribute=False)
    index = build_corpus_index([item])
    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="C")

    assert check_integrity(conditions, item, rejection) == []
    assert diag.r3_candidates_total == 1
    assert diag.r3_candidates_without_attribute == 1
    assert diag.r3_picked_letter == "D"
    touched = [i for i in range(4) if conditions["R3"].options[i].sentences != item.options[i].sentences]
    assert touched == [3]  # Dawid Kowalski: the only non-excluded option


def test_n_options_4_forced_candidate_that_carries_the_attribute_still_drops():
    """The genuinely forced case run 2 hit 444 times: the sole non-excluded option already
    carries the named attribute, so no selection can avoid gate 8 -- the item drops exactly as
    it did before this fix, not because the fix was skipped but because there was nothing to
    prefer."""
    item, rejection = make_forced_single_candidate_item(d_carries_attribute=True)
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable) as exc_info:
        build_conditions_with_diagnostics(item, rejection, index, choice_letter="C")
    exc = exc_info.value
    assert exc.gate_failures is not None
    assert any(f.startswith("gate8:") for f in exc.gate_failures)


def make_six_option_item(all_candidates_carry_attribute: bool) -> tuple[Item, Rejection]:
    """6 options: gold (A), rival (B, missing date_of_birth), the model's chosen distractor
    (C, carries date_of_birth for R1 and a separate "occupation" sentence for R2), and three
    non-excluded R3/R4 candidates in option order (D, E, F). When
    `all_candidates_carry_attribute` is False, D carries date_of_birth already and E/F do not --
    the case a first-available rule would have handed to D, forcing gate 8 to fire by luck,
    which the new selection must avoid by picking E instead. When True, D/E/F all carry it, so
    no candidate qualifies and the item must still drop via gate 8, same as today."""
    if all_candidates_carry_attribute:
        d, e, f = (
            Entity("Dawid Kowalski", ["Dawid Kowalski was born in 1930."]),
            Entity("Elena Sikorska", ["Elena Sikorska was born in 1931."]),
            Entity("Feliks Nowak", ["Feliks Nowak was born in 1932."]),
        )
    else:
        d, e, f = (
            Entity("Dawid Kowalski", ["Dawid Kowalski was born in 1928."]),
            Entity("Elena Sikorska", ["Elena Sikorska lives in Warsaw."]),
            Entity("Feliks Nowak", ["Feliks Nowak worked as a translator."]),
        )
    item = Item(
        item_id="six1",
        question="When was Bruno Kowalski born?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer."]),
            Entity(
                "Clara Novak",
                ["Clara Novak was born in 1925.", "Clara Novak worked as an editor."],
            ),
            d, e, f,
        ],
    )
    rejection = Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B", title="Bruno Kowalski", matched_by="title", attribute="date_of_birth",
        cue="birth", negation_cue="never",
    )
    return item, rejection


def test_six_option_item_prefers_the_candidate_that_lacks_the_attribute():
    item, rejection = make_six_option_item(all_candidates_carry_attribute=False)
    index = build_corpus_index([item])
    conditions, diag = build_conditions_with_diagnostics(item, rejection, index, choice_letter="C")

    assert check_integrity(conditions, item, rejection) == []
    # D (index 3) carries the attribute already; E (index 4) is the first of the two that
    # don't, so it is picked over F even though both would have worked.
    assert diag.r3_candidates_total == 3
    assert diag.r3_candidates_without_attribute == 2
    assert diag.r3_picked_letter == "E"
    touched = [i for i in range(6) if conditions["R3"].options[i].sentences != item.options[i].sentences]
    assert touched == [4]


def test_six_option_item_all_five_conditions_build():
    item, rejection = make_six_option_item(all_candidates_carry_attribute=False)
    index = build_corpus_index([item])
    conditions = build_conditions(item, rejection, index, choice_letter="C")
    assert set(conditions) == {"R0", "R1", "R2", "R3", "R4"}
    assert check_integrity(conditions, item, rejection) == []
    for label in ("R1", "R2"):
        assert conditions[label].options[1].sentences != item.options[1].sentences  # rival edited
    for label in ("R3", "R4"):
        assert conditions[label].options[4].sentences != item.options[4].sentences  # Elena edited


def test_six_option_item_falls_back_and_still_drops_when_no_candidate_qualifies():
    item, rejection = make_six_option_item(all_candidates_carry_attribute=True)
    index = build_corpus_index([item])
    with pytest.raises(RepairUnavailable) as exc_info:
        build_conditions_with_diagnostics(item, rejection, index, choice_letter="C")
    exc = exc_info.value
    assert exc.gate_failures is not None
    assert any(f.startswith("gate8:") for f in exc.gate_failures)


def test_r3_target_selection_is_deterministic():
    item, rejection = make_six_option_item(all_candidates_carry_attribute=False)
    index = build_corpus_index([item])
    first, diag1 = build_conditions_with_diagnostics(item, rejection, index, choice_letter="C")
    second, diag2 = build_conditions_with_diagnostics(item, rejection, index, choice_letter="C")
    assert diag1 == diag2
    for label in ("R0", "R1", "R2", "R3", "R4"):
        assert [o.sentences for o in first[label].options] == [
            o.sentences for o in second[label].options
        ]


# ------------------------------------------------------------------------------- --dry-run


def test_dry_run_end_to_end_produces_a_summary(tmp_path):
    from harness.run_experiment import main

    rc = main(["--dry-run", "--limit", "4", "--out-dir", str(tmp_path)])
    assert rc == 0

    import json

    summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["seed"] and summary["temperature"] == 0.0
    assert "stub" in summary["per_model"]
    assert "stage1" in summary["per_model"]["stub"]


def test_self_check_passes():
    from harness.run_experiment import self_check

    assert self_check() == 0


# --------------------------------------------------------- stage 3 records the edited letter


def test_stage3_records_the_edited_letter_and_its_own_correctness_per_condition(tmp_path):
    """`run_stage3` must add `edited_letter`/`chosen_is_edited` per condition record, without
    disturbing `chosen_is_repaired_rival` (other analyses read that field as-is, and run 1's
    records must stay comparable). Uses `make_r4_item()`: rival B, R3/R4 target C (Grazyna)."""
    import json

    from harness.models import StubBackend
    from harness.run_experiment import CONDITION_LABELS, run_stage3

    item, rejection = make_r4_item()
    index = build_corpus_index([item])
    conditions = build_conditions(item, rejection, index, choice_letter="A")
    kept = {item.item_id: (item, rejection, conditions)}

    # One call to backend.generate() per condition, in R0/R1/R2/R3/R4 order -- scripted so each
    # condition's choice is known in advance and deliberately not always the edited option.
    scripted = ["A", "B", "A", "C", "D"]  # R0, R1, R2, R3, R4

    class _Scripted:
        def __init__(self, letters):
            self.letters = list(letters)

        def __call__(self, _index, _chat):
            return f"The answer is {self.letters.pop(0)}."

    backend = StubBackend(_Scripted(scripted), name="scripted")
    outcomes, counts = run_stage3(backend, kept, tmp_path)

    assert counts["n_items"] == 1
    rows = [
        json.loads(l)
        for l in (tmp_path / "stage3_scripted.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_condition = {r["condition"]: r for r in rows}
    assert set(by_condition) == set(CONDITION_LABELS)

    expected = {
        # condition: (choice, edited_letter, chosen_is_edited, chosen_is_repaired_rival)
        "R0": ("A", None, None, False),   # nothing edited under R0
        "R1": ("B", "B", True, True),     # chose the repaired rival
        "R2": ("A", "B", False, False),   # did not move to the rival
        "R3": ("C", "C", True, False),    # chose R3's edited option, not the rival
        "R4": ("D", "C", False, False),   # chose neither the rival nor R4's edited option
    }
    for cond, (choice, edited_letter, chosen_is_edited, chosen_is_repaired_rival) in expected.items():
        row = by_condition[cond]
        assert row["choice"] == choice, cond
        assert row["edited_letter"] == edited_letter, cond
        assert row["chosen_is_edited"] == chosen_is_edited, cond
        assert row["chosen_is_repaired_rival"] == chosen_is_repaired_rival, cond
        assert row["rival_letter"] == "B", cond


# --------------------------------------------------------------- no stray control characters


def test_sources_contain_no_stray_control_characters():
    """A patch once wrote a literal backspace where the regex needed a word boundary.

    The pattern still compiled, matched nothing, and silently zeroed a whole measurement. Cheap
    to check for, invisible to read for -- see test_extract.py's identical guard.
    """
    import glob

    offenders = []
    for path in glob.glob("harness/*.py") + glob.glob("tests/*.py"):
        for i, byte in enumerate(open(path, "rb").read()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((path, i, byte))
    assert offenders == []
