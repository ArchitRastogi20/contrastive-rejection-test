"""Tests for harness.audit_instrument_defects: the non-trivial classification rules behind the
five recomputed instrument-defect figures.

Hand-written fixtures only; no GPU or network. The rules that only read committed JSONL (figures
3, 4, 5) are exercised here directly; the two rules that additionally need the R1-R4 reconstructed
sentences (figures 1, 2) are exercised on their pure helper functions instead of the full
dataset-dependent pipeline, exactly as `test_repair.py` tests `retarget`/`check_integrity` in
isolation from a real corpus.
"""

from __future__ import annotations

from harness import extract
from harness.audit_instrument_defects import (
    ABSENCE_CUES,
    RANK_ONLY_CUES,
    _enclosing_word,
    _mentions_co_candidate,
    cue_is_rank_only,
    cue_occurrence_is_embedded,
    figure3_parentage_cue,
    figure4_rejection_cue,
)


# --------------------------------------------------------------------------- cue partition


def test_rank_only_and_absence_cues_partition_negation_cues_exactly():
    """Every cue in extract.NEGATION_CUES lands in exactly one of the two sets -- this is a
    relabelling of the existing cue list, not a new vocabulary."""
    all_cues = {c.strip() for c in extract.NEGATION_CUES}
    assert RANK_ONLY_CUES | ABSENCE_CUES == all_cues
    assert RANK_ONLY_CUES & ABSENCE_CUES == set()


def test_ruled_out_is_rank_only_not_absence():
    """"I ruled out B because it is older" reports a comparison, not a missing fact."""
    assert cue_is_rank_only("ruled out") is True


def test_a_literal_negation_is_not_rank_only():
    assert cue_is_rank_only("not") is False
    assert cue_is_rank_only("lacks") is False


def test_comparison_connectives_are_rank_only():
    for cue in ("unlike", "whereas", "rather than", "instead of"):
        assert cue_is_rank_only(cue) is True


# ------------------------------------------------------------------- embedded-cue detection


def test_cue_embedded_inside_an_unrelated_word_is_detected():
    """The exact shape of the bug: "comparison" contains "son" with no word boundary."""
    padded = " it is a comparison of two films "
    assert cue_occurrence_is_embedded(padded, "son") is True


def test_cue_standing_as_its_own_word_is_not_embedded():
    padded = " she had a son named tom "
    assert cue_occurrence_is_embedded(padded, "son") is False


def test_died_embedded_inside_studied_is_detected():
    padded = " he studied philosophy "
    assert cue_occurrence_is_embedded(padded, "died") is True


def test_died_standing_as_its_own_word_is_not_embedded():
    padded = " he died in 1990 "
    assert cue_occurrence_is_embedded(padded, "died") is False


def test_cue_not_present_is_not_embedded():
    """Defensive: a cue string that (for whatever reason) is not literally in the text at all
    must not be reported as an embedded match."""
    assert cue_occurrence_is_embedded(" nothing to see here ", "son") is False


def test_enclosing_word_reads_the_full_alphabetic_run():
    assert _enclosing_word(" it is a comparison of two films ", "son") == "comparison"
    assert _enclosing_word(" the children played ", "child") == "children"


# --------------------------------------------------------------- figure 3: parentage cue bug


def _built(item_id, attribute, cue, sentence, option_titles=("Anna Kowalska", "Bruno Nowak")):
    return {
        "part": "A", "model": "m", "item_id": item_id, "attribute": attribute,
        "rival_title": option_titles[0], "sentence": sentence, "cue": cue,
        "negation_cue": "not", "option_titles": list(option_titles),
    }


def test_figure3_counts_a_genuine_substring_false_positive_as_misclassified():
    built = [
        _built("i1", "child", "son", "I ruled out B because it is a film, not a person."),
    ]
    result = figure3_parentage_cue(built)
    assert result["parentage_built_items"] == 1
    assert result["misclassified"] == 1
    assert result["misclassified_unrelated_word"] == 1
    assert result["misclassified_grandparent_word"] == 0


def test_figure3_does_not_count_a_true_parentage_mention_as_misclassified():
    built = [
        _built("i2", "child", "son", "She had a son who became a composer."),
    ]
    result = figure3_parentage_cue(built)
    assert result["misclassified"] == 0


def test_figure3_grandparent_word_is_counted_separately_from_unrelated_words():
    built = [
        _built("i3", "father", "father", "He is not her grandfather."),
    ]
    result = figure3_parentage_cue(built)
    assert result["misclassified"] == 1
    assert result["misclassified_grandparent_word"] == 1
    assert result["misclassified_unrelated_word"] == 0


def test_figure3_does_not_flag_a_shorter_cue_embedded_in_its_own_longer_sibling_cue():
    """"children" is itself a listed cue for the "child" attribute (see extract.ATTRIBUTES), so
    a sentence matched via the shorter cue "child" landing inside "children" is not a
    misclassification -- the attribute call is still correct."""
    built = [
        _built("i4", "child", "child", "The children were not mentioned in the profile."),
    ]
    result = figure3_parentage_cue(built)
    assert result["misclassified"] == 0


def test_figure3_ignores_non_parentage_built_items():
    built = [
        _built("i5", "director", "director", "She never directed anything."),
    ]
    result = figure3_parentage_cue(built)
    assert result["parentage_built_items"] == 0
    assert result["misclassified"] == 0


# ------------------------------------------------------------- figure 4: rank-only rejection


def test_figure4_counts_rank_only_and_absence_built_items_correctly():
    built = [
        {**_built("i1", "child", "son", "s1"), "negation_cue": "ruled out",
         "model": "modelA", "part": "A"},
        {**_built("i2", "director", "director", "s2"), "negation_cue": "not",
         "model": "modelA", "part": "A"},
        {**_built("i3", "child", "son", "s3"), "negation_cue": "unlike",
         "model": "modelB", "part": "C"},
    ]
    result = figure4_rejection_cue(built)
    assert result["total_built_items"] == 3
    assert result["rank_only"] == 2
    assert result["rank_only_keys"]["A"] == {"modelA::i1"}
    assert result["rank_only_keys"]["C"] == {"modelB::i3"}
    assert result["rank_only_keys"]["B"] == set()
    assert result["unclassified_cues"] == []


# ----------------------------------------------------------- figure 2 helper: co-candidate


def test_mentions_co_candidate_detects_a_distinctive_surname():
    titles = ["Anna Wilson", "Bruno Kaminski"]
    sentence = "Her brother Robert Wilson was a composer."
    assert _mentions_co_candidate(sentence, titles, edited_idx=1) is True


def test_mentions_co_candidate_ignores_a_sentence_naming_no_one():
    titles = ["Anna Wilson", "Bruno Kaminski"]
    sentence = "She was born in 1950 in Krakow."
    assert _mentions_co_candidate(sentence, titles, edited_idx=1) is False


def test_mentions_co_candidate_ignores_a_short_token_below_the_four_character_floor():
    """A one- or two-letter overlap ("de", "II") is not distinctive enough to count as naming
    someone -- the same floor `extract._distinctive_tokens` already applies."""
    titles = ["Anna De", "Bruno Kaminski"]
    sentence = "de facto this changes nothing."
    assert _mentions_co_candidate(sentence, titles, edited_idx=1) is False


def test_mentions_co_candidate_ignores_the_edited_options_own_name():
    """Mentioning the option's own name in its own inserted sentence is not a co-candidate
    mention -- only a *different* option's distinctive token counts."""
    titles = ["Anna Wilson", "Bruno Kaminski"]
    sentence = "Anna Wilson was born in 1950."
    assert _mentions_co_candidate(sentence, titles, edited_idx=0) is False
