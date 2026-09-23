from harness.audit_confound_bounds import states_parentage, stratum_flags


def test_states_parentage_is_whole_word():
    assert states_parentage("She was the daughter of a judge.")
    # the substring rule this replaces counted all three of these as parentage
    assert not states_parentage("Robinson scored 4 points in the season.")
    assert not states_parentage("A person of note.")
    assert not states_parentage("His grandmother sang.")


def test_stratum_flags():
    entry = {
        "option_titles": ["Anna Berg", "Bruno Keller", "Carla Stein"],
        "R1_sentence": "Anna Berg died in 1950, a year after Keller.", "R1_edited_idx": 0,
        "R2_sentence": "Anna Berg was the son of a farmer.", "R2_edited_idx": 0,
    }
    f = stratum_flags(entry, "date_of_death", "R1", "R2")
    assert f["no_co_candidate"] is False      # R1 names Bruno Keller's distinctive token
    assert f["template_matched"] is False     # only R2 states parentage
    assert f["date_cue_clean"] is True        # "died" is a whole word
    assert f["all_three"] is False
    entry["R1_sentence"] = "Anna Berg studied 1950 in Vienna."
    entry["R2_sentence"] = "Anna Berg sang in Vienna."
    f = stratum_flags(entry, "date_of_death", "R1", "R2")
    assert f["date_cue_clean"] is False       # "died" only inside "studied"
    assert f["no_co_candidate"] and f["template_matched"]
