"""Tests for harness.audit_date_correctness: the deterministic rules that decide whether an
inserted date makes the edited option correct under the question. Hand-written fixtures only;
no dataset, no GPU, no network."""

from __future__ import annotations

from harness.audit_date_correctness import (
    classify,
    fisher_exact,
    options_named_in,
    question_attribute,
    question_direction,
    verdict,
    year_in_profile,
    year_in_sentence,
)


def test_direction_reads_first_earlier_later_and_refuses_ambiguity():
    assert question_direction("Who was born first, A or B?") == "earlier"
    assert question_direction("Which film has the director who died earlier, A or B?") == "earlier"
    assert question_direction("Who died later, A or B?") == "later"
    assert question_direction("Which film was released more recently, A or B?") == "later"
    assert question_direction("Who is the father of A?") is None
    assert question_direction("Who was born first and died later?") is None


def test_latest_does_not_match_as_later_by_substring():
    assert question_direction("Which film came out latest, A or B?") == "later"
    # "first" and "latest" together point both ways: refused rather than guessed.
    assert question_direction("Which came out first or latest?") is None


def test_question_attribute():
    assert question_attribute("Who was born first, A or B?") == "date_of_birth"
    assert question_attribute("Who died first, A or B?") == "date_of_death"
    assert question_attribute("Which film came out first, A or B?") is None


def test_options_named_in_question_ignores_disambiguators_and_partial_words():
    titles = ["A Day", "H Day", "Remember the Day (album)", "Cast Up by the Sea"]
    q = "Which film has the director died earlier, Remember The Day or Cast Up By The Sea?"
    assert options_named_in(q, titles) == [2, 3]
    # "Day" alone must not match "A Day" inside "Remember the Day".
    assert 0 not in options_named_in(q, titles)


def test_years_from_sentence_and_profile():
    assert year_in_sentence("X was born on 3 June 1901 in Berlin.") == 1901
    assert year_in_sentence("X is a Danish footballer.") is None
    assert year_in_profile("Anna Smith (12 May 1901 - 3 June 1970) was a painter.", "date_of_birth") == 1901
    assert year_in_profile("Anna Smith (12 May 1901 - 3 June 1970) was a painter.", "date_of_death") == 1970
    assert year_in_profile("Anna Smith (12 May 1901 – 3 June 1970) was a painter.", "date_of_death") == 1970
    assert year_in_profile("Smith was born in 1888. He died in Rome in 1950.", "date_of_death") == 1950
    assert year_in_profile("Smith was born in 1888. He died in Rome in 1950.", "date_of_birth") == 1888
    assert year_in_profile("Smith was a painter from Rome.", "date_of_birth") is None


def test_verdict_direction_and_tie():
    assert verdict("earlier", 1900, 1950) == "becomes correct"
    assert verdict("earlier", 1960, 1950) == "stays incorrect"
    assert verdict("later", 1960, 1950) == "becomes correct"
    assert verdict("later", 1950, 1950).startswith("undeterminable")


def _entry(question, titles, profiles, sentence, edited_idx):
    return {"question": question, "option_titles": titles, "profiles": profiles,
            "R1_sentence": sentence, "R1_edited_idx": edited_idx}


def test_classify_scores_a_direct_person_comparison():
    e = _entry("Who died first, Anna Smith or Bob Jones?",
               ["Anna Smith", "Bob Jones", "Carl Lee", "Dora King"],
               ["Anna Smith was a painter.", "Bob Jones (1 Jan 1900 - 2 Feb 1960) was a poet.", "", ""],
               "Anna Smith died on 4 March 1955 in Paris.", 0)
    c = classify(e, "R1", "date_of_death")
    assert c["class"] == "scored"
    assert c["inserted_year"] == 1955 and c["other_year"] == 1960
    assert c["verdict"] == "becomes correct"


def test_classify_refuses_related_entity_and_unnamed_option():
    titles = ["A Day", "H Day", "Remember the Day (album)", "Cast Up by the Sea"]
    e = _entry("Which film has the director died earlier, Remember The Day or Cast Up By The Sea?",
               titles, ["", "", "", ""], "Remember the Day (album) died in 1950.", 2)
    assert classify(e, "R1", "date_of_death")["class"] == "question compares a related entity's date"
    e = _entry("Who died first, Anna Smith or Bob Jones?",
               ["Anna Smith", "Bob Jones", "Carl Lee", "Dora King"],
               ["", "Bob Jones (1900 - 1960) was a poet.", "", ""], "Carl Lee died in 1950.", 2)
    assert classify(e, "R1", "date_of_death")["class"] == "edited option not named in question"
    e = _entry("Who is the father of Anna Smith?", ["Anna Smith", "Bob Jones", "C", "D"],
               ["", "", "", ""], "Anna Smith died in 1950.", 0)
    assert classify(e, "R1", "date_of_death")["class"] == "outside stratum"


def test_fisher_exact_matches_known_table():
    # Classic tea-tasting table [[3,1],[1,3]]: two-sided p = 0.4857.
    assert abs(fisher_exact(3, 1, 1, 3) - 0.4857) < 1e-3
    assert fisher_exact(0, 0, 0, 0) == 1.0
    assert fisher_exact(5, 5, 5, 5) == 1.0
