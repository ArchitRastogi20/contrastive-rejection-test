"""Tests for harness.extract: sentence splitting, rejection/attribute extraction, and choice parsing."""

from harness.data import Entity, Item
from harness.extract import (
    analyse,
    complaint_is_true,
    find_attribute,
    find_negation,
    is_usable,
    parse_choice,
    source_of_repair,
    split_sentences,
)


def make_item() -> Item:
    return Item(
        item_id="t1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["She was born in 1970.", "She directed Blue River."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a cinematographer."]),
            Entity("Clara Novak", ["Her mother was Maria Novak."]),
        ],
    )


# ---------------------------------------------------------------- sentence splitting


def test_split_does_not_break_on_abbreviations():
    text = "The answer is Dr. Anna Kowalska. She directed it."
    assert split_sentences(text) == [
        "The answer is Dr. Anna Kowalska.",
        "She directed it.",
    ]


def test_split_handles_initials_and_newlines():
    out = split_sentences("Answer: A) J. R. Smith.\nHe directed it.")
    assert out == ["Answer: A) J. R. Smith.", "He directed it."]


# --------------------------------------------------------------------- reading a choice


def test_choice_from_named_title():
    assert parse_choice("Answer: A) Anna Kowalska.\nShe directed it.", make_item()) == "A"


def test_choice_from_bare_letter():
    assert parse_choice("The answer is B.\nIt matches.", make_item()) == "B"


def test_choice_from_option_phrasing():
    assert parse_choice("Option C is correct.", make_item()) == "C"


def test_choice_unreadable_is_none():
    assert parse_choice("It is hard to say from these profiles.", make_item()) is None


def make_six_option_item() -> Item:
    """Part C's items present six options (A-F); the letter regexes must scale with the
    item's own option count rather than the four-option case everywhere else in this file."""
    return Item(
        item_id="t2",
        question="Which film is this?",
        answer="The Face of Fu Manchu",
        gold_title="The Face of Fu Manchu",
        options=[Entity(f"Film {letter}", [f"Film {letter} plot."]) for letter in "ABCDEF"],
    )


def test_choice_from_bare_letter_past_d_on_a_six_option_item():
    # Regression guard for the bug this project shipped: `parse_choice`'s letter regexes were
    # hard-capped at [A-D], so a six-option item's model answer of "E" or "F" was invisible --
    # `choice` came back `None` even though the response plainly named a candidate.
    assert parse_choice("The answer is E.\nIt fits best.", make_six_option_item()) == "E"
    assert parse_choice("Option F is correct.", make_six_option_item()) == "F"


def make_four_option_item() -> Item:
    """Parts A and B's shape: four options, A-D."""
    return Item(
        item_id="t3",
        question="Which film is this?",
        answer="Film D",
        gold_title="Film D",
        options=[Entity(f"Film {letter}", [f"Film {letter} plot."]) for letter in "ABCD"],
    )


def test_choice_does_not_match_a_spurious_letter_past_the_item_own_option_count():
    # The other direction of the same bug class: a four-option item must never match a letter
    # past D, even if "E" appears in the text for an unrelated reason.
    item = make_four_option_item()
    assert parse_choice("Option E is correct.", item) is None
    assert parse_choice("The answer is D.\nIt matches.", item) == "D"


# ---------------------------------------- regression: rejection language is not a choice


def test_first_line_answer_beats_a_later_named_rejection():
    # This is the confirmed production bug, reproduced with make_item()'s roster: the response
    # opens by naming its choice, then restates the rejected rival's own title while ruling it
    # out ("Bruno Kowalski's profile ... I ruled out candidate B"). That second name is what
    # made the old three-line-head test see two names and fall through to `_LETTER_PICK`, which
    # then matched "candidate B" -- the rival, not the pick. The first-line rule short-circuits
    # before that fallback is ever reached.
    item = make_item()
    text = (
        "A) Anna Kowalska\n\n"
        "I chose Anna Kowalska because her profile states she directed the film. "
        "Bruno Kowalski's profile never mentions directing, so I ruled out candidate B."
    )
    assert parse_choice(text, item) == "A"


def test_first_line_answer_respects_the_item_option_count():
    item = make_six_option_item()
    text = "F) Film F\n\nI ruled out candidate B because it lacks the detail."
    assert parse_choice(text, item) == "F"


def test_letter_pick_skips_a_match_preceded_by_a_rejection_cue():
    # Without the negation guard, the *first* whole-response match of `_LETTER_PICK` is
    # "candidate B" -- right after "ruled out" -- so a bare `.search()` would return B even
    # though the model's actual pick, stated afterwards, is A.
    item = make_item()
    text = (
        "I ruled out candidate B because nothing in the profile confirms he ever directed "
        "a film. Based on the profiles, the correct choice is A."
    )
    assert parse_choice(text, item) == "A"


def test_letter_pick_guard_does_not_reach_into_an_unrelated_earlier_clause():
    # The guard's left-context window must not be so wide that an earlier, unrelated negation
    # -- well clear of the matched keyword -- suppresses a later, genuine pick.
    item = make_item()
    text = (
        "I want to note that the profile for Bruno does not mention anything about "
        "directing at all, which is a shame given how promising his other work looked. "
        "Setting all of that aside, the correct option is A."
    )
    assert parse_choice(text, item) == "A"


def test_rejection_still_extracted_when_first_line_rule_fixes_the_choice():
    # Before the fix, this exact text parsed to choice "B" (the rejected rival, via the same
    # mechanism as the confirmed bug), and `analyse` then silently dropped the only rejection in
    # the text -- its loop skips a rejection of whichever letter equals `choice`. Fixing the
    # choice also restores the rejection: it was never a broken rejection regex, only a wrong
    # `choice` making a genuine rejection look like "the model rejected its own pick".
    item = make_item()
    text = (
        "A) Anna Kowalska\n\n"
        "I chose Anna Kowalska because her profile states she directed the film. "
        "Bruno Kowalski's profile never mentions directing, so I ruled out option B for this one."
    )
    a = analyse(text, item)
    assert a.choice == "A"
    assert len(a.rejections) == 1
    rej = a.rejections[0]
    assert rej.letter == "B"
    assert rej.title == "Bruno Kowalski"
    assert rej.attribute == "director"


# ------------------------------------------------------------------------- attributes


def test_attribute_and_negation_cues():
    assert find_attribute("his profile never gives a date of birth")[0] == "date_of_birth"
    assert find_attribute("no mention of his mother")[0] == "mother"
    assert find_attribute("this one is simply less relevant")[0] is None
    assert find_negation("his profile never gives a date") == "never"
    assert find_negation("this profile matches the question") is None


def test_two_letter_cue_needs_a_boundary():
    # "b." must not fire inside an ordinary word such as "Bob." or "club."
    assert find_attribute("he was a member of the club.")[0] is None


# ------------------------------------------------------------------- whole-response read


def test_specific_rejection_is_found_and_scored():
    item = make_item()
    text = (
        "Answer: A) Anna Kowalska.\n"
        "Anna Kowalska directed the film. "
        "It is not Bruno Kowalski, because his profile never gives a date of birth."
    )
    a = analyse(text, item)
    assert a.choice == "A"
    assert len(a.rejections) == 1

    rej = a.rejections[0]
    assert rej.title == "Bruno Kowalski"
    assert rej.attribute == "date_of_birth"
    # the complaint is true: Bruno's profile has no birth information
    assert complaint_is_true(item, rej) is True
    # and it is repairable from released text: Anna's profile carries a birth date
    assert source_of_repair(item, rej).title == "Anna Kowalska"
    assert is_usable(item, rej) is True


def test_vague_rejection_is_not_usable():
    item = make_item()
    text = "Answer: A) Anna Kowalska.\nBruno Kowalski is not relevant here."
    rej = analyse(text, item).rejections[0]
    assert rej.attribute is None
    assert rej.is_specific is False
    assert is_usable(item, rej) is False


def test_rejection_of_the_chosen_option_is_not_counted():
    item = make_item()
    text = "Answer: B) Bruno Kowalski.\nBruno Kowalski is not a director, but he is the closest."
    assert analyse(text, item).rejections == []


def test_false_complaint_is_flagged():
    item = make_item()
    # Clara's profile does mention a mother, so this complaint is wrong
    text = "Answer: A) Anna Kowalska.\nClara Novak's profile never names her mother."
    rej = analyse(text, item).rejections[0]
    assert rej.attribute == "mother"
    assert complaint_is_true(item, rej) is False
    assert is_usable(item, rej) is False


def test_surname_alone_identifies_an_option():
    item = make_item()
    text = "Answer: A) Anna Kowalska.\nNovak lacks any mention of directing."
    rej = analyse(text, item).rejections[0]
    assert rej.title == "Clara Novak"
    assert rej.matched_by == "token"


# ------------------------------------- the three defects the first pilot run exposed


def test_a_bare_date_complaint_is_not_repairable():
    """"No date is given" names nothing anyone can supply -- which date?

    This bucket existed in the first run, fired on 23 rejections, and was wrong at both ends:
    it counted as specific, and its presence check looked for the literal word "date" in the
    profile, which a Wikipedia sentence never contains even when it states one.
    """
    assert find_attribute("no date is given for this candidate")[0] is None
    assert find_attribute("the year is not stated anywhere")[0] is None
    # naming *which* date still works
    assert find_attribute("no date of birth is given")[0] == "date_of_birth"


def test_born_without_a_date_is_not_a_date_of_birth():
    from harness.extract import attribute_in_profile

    assert attribute_in_profile("She was born in 1970 in Krakow.", "date_of_birth") is True
    assert attribute_in_profile("She was born in Krakow.", "date_of_birth") is False
    assert attribute_in_profile("He died in March 1994.", "date_of_death") is True
    assert attribute_in_profile("He died in Milan.", "date_of_death") is False
    # a non-dated attribute is unaffected by the date requirement
    assert attribute_in_profile("Her mother was Maria.", "mother") is True


def test_a_candidate_name_cannot_supply_the_attribute_cue():
    from harness.extract import strip_titles

    titles = ["Robert Bresson", "When Were You Born", "Charles Saunders (director)"]
    # "Bres-son" supplied "son", "When Were You Born" supplied "born", and the parenthetical
    # supplied "director" in the first run: ten rejections were misclassified
    assert find_attribute("Robert Bresson is not relevant")[0] == "child"
    assert find_attribute(strip_titles("Robert Bresson is not relevant", titles))[0] is None
    assert find_attribute(strip_titles("When Were You Born lacks detail", titles))[0] is None


def test_sources_contain_no_stray_control_characters():
    """A patch once wrote a literal backspace where the regex needed a word boundary.

    The pattern still compiled, matched nothing, and silently zeroed a whole measurement. Cheap
    to check for, invisible to read for.
    """
    import glob

    offenders = []
    for path in glob.glob("harness/*.py") + glob.glob("tests/*.py"):
        for i, byte in enumerate(open(path, "rb").read()):
            if byte < 9 or byte in (11, 12) or 14 <= byte < 32:
                offenders.append((path, i, byte))
    assert offenders == []
