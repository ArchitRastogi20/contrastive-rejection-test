"""Loader and item construction. The three context layouts are the ones 2Wiki ships in."""

import pytest

from pilot.config import FIXTURE_DIR, N_OPTIONS, SEED
from pilot.data import (
    build_item,
    build_items,
    describe_schema,
    load_records_from_file,
    parse_context,
    parse_evidences,
)

FIXTURE = FIXTURE_DIR / "twowiki_sample.jsonl"


# ------------------------------------------------------------------------ context shapes


def test_context_as_pairs():
    ents = parse_context([["A", ["one.", "two."]], ["B", ["three."]]])
    assert [e.title for e in ents] == ["A", "B"]
    assert ents[0].profile == "one. two."


def test_context_as_dicts():
    ents = parse_context([{"title": "A", "sentences": ["one."]}])
    assert ents[0].title == "A" and ents[0].profile == "one."


def test_context_as_columns():
    ents = parse_context({"title": ["A", "B"], "sentences": [["one."], ["two."]]})
    assert [e.title for e in ents] == ["A", "B"]


def test_unrecognised_context_is_loud():
    with pytest.raises(ValueError):
        parse_context(42)


def test_evidences_tolerate_both_shapes():
    assert parse_evidences([["a", "r", "b"]]) == [("a", "r", "b")]
    assert parse_evidences([{"subject": "a", "relation": "r", "object": "b"}]) == [
        ("a", "r", "b")
    ]
    assert parse_evidences(None) == []


# --------------------------------------------------------------------- item construction


def test_builds_an_item_from_the_fixture():
    records = load_records_from_file(FIXTURE)
    item = build_item(records[0], n_options=N_OPTIONS, seed=SEED)
    assert item is not None
    assert item.gold_title == "Anna Kowalska"
    assert len(item.options) == N_OPTIONS
    assert item.options[ord(item.gold_letter) - ord("A")].title == "Anna Kowalska"


def test_value_answers_are_skipped():
    # fix-0004's answer is a year, not one of the candidate entities
    records = {r["_id"]: r for r in load_records_from_file(FIXTURE)}
    assert build_item(records["fix-0004"], n_options=N_OPTIONS, seed=SEED) is None


def test_option_order_is_deterministic():
    records = load_records_from_file(FIXTURE)
    first = build_item(records[0], n_options=N_OPTIONS, seed=SEED)
    again = build_item(records[0], n_options=N_OPTIONS, seed=SEED)
    assert [o.title for o in first.options] == [o.title for o in again.options]


def test_type_match_score_rewards_shared_title_tokens():
    records = {r["_id"]: r for r in load_records_from_file(FIXTURE)}
    # every Ferri distractor shares a surname with the gold entity
    ferri = build_item(records["fix-0002"], n_options=3, seed=SEED)
    # no Press name shares a token with "Northwind Press" except the generic word "press",
    # which is not a stopword, so this one also scores -- the point is it is recorded, not
    # that it is high
    press = build_item(records["fix-0003"], n_options=3, seed=SEED)
    assert ferri.type_match_score == 1.0
    assert 0.0 <= press.type_match_score <= 1.0


def test_build_items_stops_at_the_limit():
    records = load_records_from_file(FIXTURE)
    assert len(build_items(records, n_items=2, n_options=N_OPTIONS, seed=SEED)) == 2


def test_schema_description_mentions_every_field():
    described = describe_schema(load_records_from_file(FIXTURE))
    for field in ("question", "answer", "context", "evidences"):
        assert field in described
