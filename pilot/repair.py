"""Construct the five repair conditions (R0-R4) from an item's contrastive rejection.

A deterministic, pure transform on strings and dataclasses: given the same item, rejection and
corpus index, every function returns byte-identical output (see the experiment design doc). The
repair takes a real sentence from the corpus carrying the attribute the model said was missing,
retargets its subject onto the rival, and inserts it into the rival's profile. R4 completes the
2x2 of content (relevant/irrelevant) crossed with location (rival/third option), reusing R2's
source sentence retargeted onto R3's target.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from . import extract
from . import config as C
from .data import Entity, Item, _norm, _title_tokens, _WORD
from .extract import ATTRIBUTES, Rejection

# ----------------------------------------------------------------------------- exceptions


class RepairUnavailable(Exception):
    """The repair cannot be mechanically constructed for this item.

    Raised, never worked around: the caller counts the drop and records this message as the
    reason. See the experiment design doc's repair section on the residual that is dropped
    rather than patched.

    `gate_failures` and `cap_hit` are optional extra detail for the caller's audit trail: which
    gates kept failing on the last combination tried, and whether the search gave up because it
    hit `R1_SEARCH_CAP`/`R2_SEARCH_CAP` rather than because it ran out of real candidates.
    """

    def __init__(
        self, message: str, *, gate_failures: list[str] | None = None, cap_hit: bool = False
    ) -> None:
        super().__init__(message)
        self.gate_failures = gate_failures
        self.cap_hit = cap_hit


# ------------------------------------------------------------------------- locating a sentence


def find_attribute_sentence(entity: Entity, attribute: str) -> str | None:
    """The one sentence in `entity`'s profile that carries `attribute`, or None.

    Reuses the extractor's own cue matching (`find_attribute`) and its own date-token rule
    (`_DATE_TOKEN`, `_DATED_ATTRIBUTES`) so a sentence judged to "carry date_of_birth" here is
    judged by exactly the rule `attribute_in_profile` uses -- a profile that only says "born in
    Krakow" does not carry date_of_birth, because the sentence has no date in it.
    """
    for sentence in extract.split_sentences(entity.profile):
        stripped = extract.strip_titles(sentence, [entity.title])
        attr, _cue = extract.find_attribute(stripped)
        if attr != attribute:
            continue
        if attribute in extract._DATED_ATTRIBUTES and not extract._DATE_TOKEN.search(sentence):
            continue
        return sentence.strip()
    return None


def _token_len(text: str) -> int:
    return len(_WORD.findall(text))


_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def _years(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(y) for y in _YEAR_RE.findall(text)]


# ------------------------------------------------------------------------------ retargeting

# Measured on the pilot's own rejections: the sentence names its own subject 50% of the time,
# opens with one of these pronouns 1.5% more, and the rest is dropped.
_PRONOUNS = ("He", "She", "It", "They", "His", "Her", "Its", "Their")
_LEADING_PRONOUN = re.compile(r"^(" + "|".join(_PRONOUNS) + r")\b(.*)$", re.IGNORECASE | re.DOTALL)
_LEADING_COPULA = re.compile(r"^(\s+)(was|were|is|are)\b", re.IGNORECASE)
_COPULA_SINGULAR = {"was": "was", "were": "was", "is": "is", "are": "is"}


def _retarget_by_name(sentence: str, source_title: str, target_title: str) -> str | None:
    """Case 1: the sentence names its own subject -- substitute the name.

    Matches on the full title anywhere in the sentence, then on the title's distinctive
    tokens (a bare surname such as "Kowalski" standing in for "Bruno Kowalski") -- but only
    when that token opens the sentence. A token match anywhere in the sentence is not safe:
    "Her mother was Maria Novak." contains the token "Novak" but it names the *mother*, not
    the sentence's subject, and the subject here is the leading pronoun "Her". Restricting the
    token match to the leading position is what keeps that case for `_retarget_by_pronoun`.
    """
    whole = re.compile(re.escape(source_title), re.IGNORECASE)
    if whole.search(sentence):
        return whole.sub(target_title, sentence, count=1)

    for token in sorted(_title_tokens(source_title), key=lambda t: (-len(t), t)):
        leading = re.compile(rf"^(\s*){re.escape(token)}\b", re.IGNORECASE)
        m = leading.search(sentence)
        if m:
            return leading.sub(lambda mm: mm.group(1) + target_title, sentence, count=1)
    return None


def _retarget_by_pronoun(sentence: str, target_title: str) -> str | None:
    """Case 2: the sentence opens with a pronoun -- replace it with the target's name.

    Verb agreement is fixed only where the pronoun is immediately followed by a copula
    (was/were/is/are): "He was born" -> "<Name> was born". A possessive such as "His mother
    was Maria" is not followed by a verb at all, so nothing downstream is touched -- the design
    accepts this as mechanical and imperfect; it is 1.5% of cases.
    """
    m = _LEADING_PRONOUN.match(sentence)
    if not m:
        return None
    remainder = m.group(2)
    remainder = _LEADING_COPULA.sub(
        lambda cm: cm.group(1) + _COPULA_SINGULAR[cm.group(2).casefold()],
        remainder,
        count=1,
    )
    return target_title + remainder


def retarget(sentence: str, source_title: str, target_title: str) -> str | None:
    """Replace `sentence`'s subject with `target_title`, or return None if neither mechanical
    case applies. Callers must drop the item on None rather than inventing a third case."""
    by_name = _retarget_by_name(sentence, source_title, target_title)
    if by_name is not None:
        return by_name
    return _retarget_by_pronoun(sentence, target_title)


# --------------------------------------------------------------------------- the corpus index


def build_corpus_index(items: list[Item]) -> dict:
    """One pass over every option in every item: which real sentences carry which attribute,
    and what the corpus's evidence triples say about each entity's own values.

    A plain dict, built once per run and re-used across items -- no new dependency, and no
    class needed since nothing here mutates after construction.
    """
    by_attribute: dict[str, list[tuple[str, str, str]]] = {a: [] for a in ATTRIBUTES}
    evidence_values: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()

    for item in items:
        for opt in item.options:
            for attr in ATTRIBUTES:
                sentence = find_attribute_sentence(opt, attr)
                if sentence is None:
                    continue
                key = (opt.title, sentence)
                if key in seen:
                    continue
                seen.add(key)
                by_attribute[attr].append((opt.title, sentence, item.item_id))
        for subject, relation, obj in item.relations:
            attr = relation.strip().casefold().replace(" ", "_")
            if attr in ATTRIBUTES:
                evidence_values.setdefault(_norm(subject), []).append((attr, obj))

    for attr, entries in by_attribute.items():
        entries.sort(key=lambda e: (e[0], e[1], e[2]))

    return {"by_attribute": by_attribute, "evidence_values": evidence_values}


def _repair_source_candidates(item: Item, rejection: Rejection) -> list[tuple[Entity, str]]:
    """Every sibling in this item that carries the named attribute in one isolable sentence,
    in option order -- the pool `select_repair_source` chooses R1's source from."""
    candidates: list[tuple[Entity, str]] = []
    for opt in item.options:
        if _norm(opt.title) == _norm(rejection.title):
            continue
        sentence = find_attribute_sentence(opt, rejection.attribute)
        if sentence is not None:
            candidates.append((opt, sentence))
    return candidates


def _r1_candidates_ordered(
    item: Item, rejection: Rejection, corpus_index: dict
) -> list[tuple[Entity, str, str]]:
    """Every R1 candidate -- (sibling, sentence, stratum) -- in full preference order.

    True-value matches (a candidate sentence carrying the rejected entity's own real value,
    read off this corpus's evidence triples) come first, in option order; every remaining
    candidate follows, also in option order, labelled "borrowed". No candidate appears twice.

    This is the pool `select_repair_source` used to collapse into a single pick by taking the
    head; `build_conditions_with_diagnostics` walks the whole list instead, so a candidate that
    fails to retarget or fails a gate no longer drops the item -- the next one in this order is
    tried.
    """
    candidates = _repair_source_candidates(item, rejection)
    if not candidates:
        return []

    real_values = [
        v for attr, v in corpus_index.get("evidence_values", {}).get(_norm(rejection.title), [])
        if attr == rejection.attribute
    ]

    true_value: list[tuple[Entity, str, str]] = []
    borrowed: list[tuple[Entity, str, str]] = []
    for opt, sentence in candidates:
        low = sentence.casefold()
        if real_values and any(str(v).casefold() in low for v in real_values):
            true_value.append((opt, sentence, "true_value"))
        else:
            borrowed.append((opt, sentence, "borrowed"))
    return true_value + borrowed


def select_repair_source(
    item: Item, rejection: Rejection, corpus_index: dict
) -> tuple[Entity, str, str] | None:
    """The single best R1 sibling and sentence, and which stratum that choice belongs to.

    This is just the head of `_r1_candidates_ordered` -- kept as its own name because
    `classify_stratum` and the tests read it as "the" repair source. The real build
    (`build_conditions_with_diagnostics`) walks the full ordered list and may end up using a
    candidate further down it when the head is not mechanically retargetable or does not pass
    the integrity gates; this function does not know that and should not be read as "what got
    built".

    Returns None when no sibling carries the attribute in one isolable sentence at all.
    """
    ordered = _r1_candidates_ordered(item, rejection, corpus_index)
    return ordered[0] if ordered else None


def classify_stratum(item: Item, rejection: Rejection, corpus_index: dict) -> str:
    """"true_value" or "borrowed" -- see `select_repair_source`, whose label this just reads
    off. Kept as its own call so stage 2 can report the stratum without re-deriving the source
    sentence itself."""
    picked = select_repair_source(item, rejection, corpus_index)
    return picked[2] if picked is not None else "borrowed"


def _r2_candidate_pool(
    item: Item, rival_title: str, named_attribute: str, corpus_index: dict
) -> list[tuple[int, str, str, str]]:
    """Every eligible R2 (token_len, attribute, source_title, sentence) quadruple, deduplicated
    and in a canonical (attribute, title, sentence) order that does not depend on `target_len`.

    Built from this item's own siblings and from the whole corpus index (excluding the rival
    itself and the named attribute either way) -- the same two sources `_pick_r2_source` always
    drew from, just kept as a reusable pool instead of collapsed into one pick. A `set` is used
    only to drop exact duplicates; the pool is always returned `sorted()`, so which physical
    hash bucket a tuple landed in never affects the result. The token length is computed once
    here rather than inside the sort key that runs on every re-ordering.
    """
    pool: set[tuple[str, str, str]] = set()

    for opt in item.options:
        if _norm(opt.title) == _norm(rival_title):
            continue
        for attr in sorted(ATTRIBUTES):
            if attr == named_attribute:
                continue
            sentence = find_attribute_sentence(opt, attr)
            if sentence is not None:
                pool.add((attr, opt.title, sentence))

    for attr, entries in corpus_index.get("by_attribute", {}).items():
        if attr == named_attribute:
            continue
        for title, sentence, _item_id in entries:
            if _norm(title) == _norm(rival_title):
                continue
            pool.add((attr, title, sentence))

    return [(_token_len(sentence), attr, title, sentence) for attr, title, sentence in sorted(pool)]


def _r2_candidates_ordered(
    pool: list[tuple[int, str, str, str]], target_len: int
) -> list[tuple[str, str, str]]:
    """The pool re-ordered for one `target_len`: closest in length first, tie-broken by
    attribute name then source title then sentence text -- the same tie-break
    `_pick_r2_source` always used, just applied to the whole pool instead of only to the min."""

    def key(c: tuple[int, str, str, str]) -> tuple[int, str, str, str]:
        length, attr, title, sentence = c
        return (abs(length - target_len), attr, title, sentence)

    return [(attr, title, sentence) for _len, attr, title, sentence in sorted(pool, key=key)]


def _pick_r2_source(
    item: Item, rival_title: str, named_attribute: str, corpus_index: dict, target_len: int
) -> tuple[str, str, str] | None:
    """(attribute, source_title, sentence) for R2: a different attribute, closest in length
    to `target_len`. The single-pick counterpart to `_r2_candidates_ordered`, kept for callers
    that only ever want the one best candidate."""
    pool = _r2_candidate_pool(item, rival_title, named_attribute, corpus_index)
    ordered = _r2_candidates_ordered(pool, target_len)
    return ordered[0] if ordered else None


# --------------------------------------------------------------------------- building an item


def _with_sentence_appended(item: Item, option_index: int, sentence: str) -> Item:
    """A deep copy of `item` with one extra sentence at the end of one option's profile.

    Appending (never inserting or rewriting) is the convention `check_integrity` relies on to
    find "the sentence this condition added" by diffing against the untouched item.
    """
    clone = copy.deepcopy(item)
    clone.options[option_index].sentences.append(sentence)
    return clone


# Caps on the joint R1 x R2 search below, so a pathological item (a huge true-value stratum,
# or an item whose rival matches a common attribute across the whole corpus) cannot spin: at
# most this many R1 candidates, and for each of those, at most this many R2 candidates. Measured
# on the global pool, 7,272 R2 candidates exist and 70% retarget cleanly, so 40 is generous
# headroom over what a real item needs, not a tight budget.
R1_SEARCH_CAP = 40
R2_SEARCH_CAP = 40


@dataclass
class RepairDiagnostics:
    """What the search had to do to build this item -- the auditable trail `build_conditions`
    itself throws away. `r1_examined`/`r2_examined` are 1-based: "the winning candidate was the
    k-th tried", not "k candidates existed". `cap_hit` is true when either search truncated a
    longer candidate list at `R1_SEARCH_CAP`/`R2_SEARCH_CAP`, whether or not that truncation
    ended up mattering for this particular item.

    `r3_candidates_total`/`r3_candidates_without_attribute`/`r3_picked_letter` record what the
    R3/R4 target selection had to work with: how many non-excluded options existed, how many of
    those already lacked the named attribute (gate 8's requirement), and which letter was
    actually picked -- always the true counts, even at `n_options<=4` where the preference for
    a lacking candidate is disabled (see `build_conditions_with_diagnostics`) and the pick is
    just the first available one. `r3_candidates_total` is 1 whenever the rival, the model's
    own stage-1 choice and the gold letter are three distinct letters (the forced-choice case
    the experiment design doc's gate 8 section measures), but is 2 whenever the model's choice was
    the gold answer itself, which only removes two distinct letters from a 4-option item -- not
    a rare edge case: 62.7% of run 2's built items were stage-1-correct. A larger option count
    is what gives this selection room, on every item rather than only that minority, to prefer
    a candidate that already satisfies gate 8 rather than being handed one of few and hoping."""

    stratum: str
    r1_examined: int
    r2_examined: int
    r1_total: int
    r2_total: int
    cap_hit: bool
    r3_candidates_total: int
    r3_candidates_without_attribute: int
    r3_picked_letter: str


def build_conditions_with_diagnostics(
    item: Item, rejection: Rejection, corpus_index: dict, choice_letter: str | None = None,
    n_options: int | None = None,
) -> tuple[dict[str, Item], RepairDiagnostics]:
    """The five conditions R0-R4, plus the search trail, or raise RepairUnavailable.

    `choice_letter` is the letter the model actually chose in stage 1 (not necessarily the
    gold letter -- see the report for why this parameter was added beyond the design doc's
    literal signature). It must be an actual letter, not defaulted: when the model's response
    could not be parsed, `choice_letter` is None and this raises rather than guessing gold.

    `n_options` is the option count `item` was built with (`config.N_OPTIONS` at the time
    `data.build_items` ran) -- threaded through explicitly rather than silently re-derived, so
    a caller is never ambiguous about it. Defaults to `len(item.options)` when not given, which
    is always correct for items `data.build_item` produced (it never returns an item with a
    different option count than it was asked for). It gates the R3/R4 target-selection
    preference described below: at 4 or fewer options it is disabled, so `build_conditions`
    reproduces runs 1 and 2 byte-for-byte (see the paragraph below on why that gate exists and
    is not merely cosmetic).

    Everything that is a property of the *item* -- the rival is not among the options, the
    rival is the gold answer, the choice is unreadable, no non-chosen non-gold option exists
    for R3/R4 -- is checked once, up front, independently of any candidate. R3/R4's target
    itself is also chosen up front, once, from among the non-excluded options. At more than 4
    options, whichever one already lacks the named attribute is preferred (a deterministic
    first-in-option-order tie-break among those that qualify), so gate 8 tends to pass by
    construction rather than by the luck of which option happened to be left over -- see
    `RepairDiagnostics` for what that selection recorded (recorded always, whether or not the
    preference is active, so the search's raw material is auditable either way). At 4 options
    or fewer, the target is simply the first non-excluded option, exactly as before this
    preference existed -- *not* because there is provably only one such option (there is not:
    when the model's stage-1 choice happens to be the gold answer, excluding {rival, choice,
    gold} removes only two distinct letters at n_options=4, leaving two candidates, not one --
    measured on run 2, this is the majority case, since 62.7% of its built items were
    stage-1-correct), but because changing which of several already-legal candidates gets
    picked changes which items build, and runs 1 and 2 are a committed scientific record that
    must stay reproducible from this code. The preference is confined to option counts strictly
    greater than the historical default so that record is never disturbed. Everything that is a
    property of the *candidate* -- retargetability, the eight integrity gates -- is searched:
    R1 candidates are tried in preference order (true-value stratum first, then the existing
    option order), and for each R1 candidate that retargets (and whose sentence also retargets
    onto R3's target), R2 candidates are tried in preference order (closest in length, same
    tie-break as before). Each R2 candidate that retargets onto the rival is also retargeted
    onto R3's target for R4 -- the same source sentence, never a second independent search --
    and the combination is accepted only once R1/R2/R3/R4 together pass every gate. The item is
    dropped only once both searches are genuinely exhausted (or the cap is hit), never on the
    first candidate that happens to fail.
    """
    rival_idx = next(
        (i for i, o in enumerate(item.options) if _norm(o.title) == _norm(rejection.title)), None
    )
    if rival_idx is None:
        raise RepairUnavailable(f"named rival {rejection.title!r} is not among this item's options")
    rival_title = item.options[rival_idx].title
    if _norm(rival_title) == _norm(item.gold_title):
        # The model rejected the gold answer itself -- a different failure mode (it also chose
        # wrong) from the one this study measures. Repairing it would mean editing the gold
        # profile, which gate 6 exists to forbid, so this item is out of scope, not a gate
        # failure to report.
        raise RepairUnavailable("named rival is the gold answer; out of scope for this study")
    if choice_letter is None:
        # `analysis.choice` is None whenever the response could not be parsed -- not
        # hypothetical, one pilot cell had 8 of 40 unreadable. Silently falling back to the
        # gold letter would let R3 land on the option the model actually chose, destroying the
        # control R3 exists to be, invisibly. Drop the item instead.
        raise RepairUnavailable(
            "model's choice is unreadable; cannot pick a non-chosen option for R3"
        )
    if n_options is None:
        n_options = len(item.options)

    excluded = {rejection.letter, choice_letter, item.gold_letter}
    r3_candidate_indices = [
        i for i in range(len(item.options)) if chr(ord("A") + i) not in excluded
    ]
    if not r3_candidate_indices:
        raise RepairUnavailable("no other non-chosen, non-gold option available for R3")

    # Which candidates already lack the named attribute -- computed and recorded regardless of
    # whether the preference below is active, so the diagnostics are always the true picture of
    # what the search had to work with, not just of what it acted on.
    all_titles = [o.title for o in item.options]
    r3_indices_without_attribute = [
        i for i in r3_candidate_indices
        if not extract.attribute_in_profile(
            item.options[i].profile, rejection.attribute, all_titles
        )
    ]

    if n_options > C.PREFER_LACKING_TARGET_ABOVE and r3_indices_without_attribute:
        # Prefer a target that already lacks the named attribute, so gate 8 is satisfied by
        # construction rather than by luck. Deterministic tie-break: first in existing option
        # order among those that qualify, the same convention `_repair_source_candidates` and
        # `_r2_candidate_pool` use elsewhere in this file. Confined to n_options > 4 -- see the
        # docstring above for why runs 1 and 2 (n_options=4) must not take this branch.
        r3_idx = r3_indices_without_attribute[0]
    else:
        # n_options <= 4 (today's default, and the two committed runs), or no candidate
        # qualifies: the first available one, exactly as before this preference existed. The
        # gate is never weakened by this fallback -- it either fires exactly as it does today,
        # or the search below finds a combination that passes it anyway.
        r3_idx = r3_candidate_indices[0]
    r3_title = item.options[r3_idx].title
    r3_candidates_total = len(r3_candidate_indices)
    r3_candidates_without_attribute = len(r3_indices_without_attribute)

    r1_ordered = _r1_candidates_ordered(item, rejection, corpus_index)
    if not r1_ordered:
        raise RepairUnavailable(
            f"no sibling profile carries {rejection.attribute} in one isolable sentence"
        )

    r1_total = len(r1_ordered)
    r1_list = r1_ordered[:R1_SEARCH_CAP]
    cap_hit = r1_total > R1_SEARCH_CAP

    r2_pool = _r2_candidate_pool(item, rival_title, rejection.attribute, corpus_index)

    r1_examined = 0
    any_r1_retargeted = False
    any_r1_and_r3_retargeted = False
    any_r2_available = False
    any_r2_retargeted = False
    any_r2_and_r4_retargeted = False
    r2_examined_last = 0
    last_gate_failures: list[str] = []
    r0 = copy.deepcopy(item)  # unmodified in every attempt; built once, not per combination

    for r1_examined, (source_entity, source_sentence, stratum) in enumerate(r1_list, start=1):
        r1_sentence = retarget(source_sentence, source_entity.title, rival_title)
        if r1_sentence is None:
            continue
        any_r1_retargeted = True
        r3_sentence = retarget(source_sentence, source_entity.title, r3_title)
        if r3_sentence is None:
            continue
        any_r1_and_r3_retargeted = True

        target_len = _token_len(r1_sentence)
        r2_total = len(r2_pool)
        r2_ordered = _r2_candidates_ordered(r2_pool, target_len)
        if r2_ordered:
            any_r2_available = True
        if r2_total > R2_SEARCH_CAP:
            cap_hit = True
        r2_list = r2_ordered[:R2_SEARCH_CAP]

        for r2_examined, (_r2_attr, r2_source_title, r2_source_sentence) in enumerate(
            r2_list, start=1
        ):
            r2_examined_last = r2_examined
            r2_sentence = retarget(r2_source_sentence, r2_source_title, rival_title)
            if r2_sentence is None:
                continue
            any_r2_retargeted = True
            # R4 reuses R2's own source sentence -- the same one just retargeted onto the
            # rival above -- retargeted onto R3's target instead. No independent search: if
            # this particular R2 candidate does not also retarget onto r3_title, the whole
            # combination is rejected and the next R2 candidate is tried, exactly like R1/R3.
            r4_sentence = retarget(r2_source_sentence, r2_source_title, r3_title)
            if r4_sentence is None:
                continue
            any_r2_and_r4_retargeted = True

            conditions = {
                "R0": r0,
                "R1": _with_sentence_appended(item, rival_idx, r1_sentence),
                "R2": _with_sentence_appended(item, rival_idx, r2_sentence),
                "R3": _with_sentence_appended(item, r3_idx, r3_sentence),
                "R4": _with_sentence_appended(item, r3_idx, r4_sentence),
            }
            failures = check_integrity(conditions, item, rejection)
            if not failures:
                return conditions, RepairDiagnostics(
                    stratum=stratum,
                    r1_examined=r1_examined,
                    r2_examined=r2_examined,
                    r1_total=r1_total,
                    r2_total=r2_total,
                    cap_hit=cap_hit,
                    r3_candidates_total=r3_candidates_total,
                    r3_candidates_without_attribute=r3_candidates_without_attribute,
                    r3_picked_letter=chr(ord("A") + r3_idx),
                )
            last_gate_failures = failures

    # Genuinely exhausted: report the most specific reason the search has evidence for, in the
    # same order a person would ask the questions -- could R1 even be built, could R3, was there
    # an R2 source at all, could it be retargeted, could R4 (the same source, onto R3's target)
    # be retargeted, and only then "every combination failed a gate". Each of these maps to one
    # of the drop categories `run_experiment._DROP_CATEGORY` already recognised, via substring
    # match on the message.
    if not any_r1_retargeted:
        raise RepairUnavailable(
            f"R1 sentence is not mechanically retargetable (tried {r1_examined} candidate(s))",
            cap_hit=cap_hit,
        )
    if not any_r1_and_r3_retargeted:
        raise RepairUnavailable(
            f"R3 sentence is not mechanically retargetable (tried {r1_examined} candidate(s))",
            cap_hit=cap_hit,
        )
    if not any_r2_available:
        raise RepairUnavailable("no alternate-attribute source available for R2", cap_hit=cap_hit)
    if not any_r2_retargeted:
        raise RepairUnavailable(
            f"R2 sentence is not mechanically retargetable (tried {r2_examined_last} "
            f"candidate(s))",
            cap_hit=cap_hit,
        )
    if not any_r2_and_r4_retargeted:
        raise RepairUnavailable(
            f"R4 sentence is not mechanically retargetable (tried {r2_examined_last} "
            f"candidate(s))",
            cap_hit=cap_hit,
        )
    raise RepairUnavailable(
        f"no R1/R2/R4 candidate combination passed every integrity gate (examined {r1_examined} "
        f"R1 x up to {r2_examined_last} R2 candidates; last failures: "
        f"{'; '.join(last_gate_failures)})",
        gate_failures=last_gate_failures,
        cap_hit=cap_hit,
    )


def build_conditions(
    item: Item, rejection: Rejection, corpus_index: dict, choice_letter: str | None = None,
    n_options: int | None = None,
) -> dict[str, Item]:
    """The five conditions R0-R4 from the experiment design doc, or raise RepairUnavailable.

    A thin wrapper around `build_conditions_with_diagnostics` that drops the search trail, for
    callers (and the existing tests) that only need the conditions themselves.
    """
    conditions, _diagnostics = build_conditions_with_diagnostics(
        item, rejection, corpus_index, choice_letter, n_options
    )
    return conditions


# ----------------------------------------------------------------------------- integrity gates


def _edited_slot(item: Item, cond_item: Item) -> tuple[int | None, list[str]]:
    """Which option's sentence list changed, and what got appended -- by diffing against the
    untouched item, per the append-only convention `_with_sentence_appended` establishes."""
    for i, (orig, new) in enumerate(zip(item.options, cond_item.options)):
        if orig.sentences != new.sentences:
            return i, new.sentences[len(orig.sentences):]
    return None, []


def check_integrity(conditions: dict[str, Item], item: Item, rejection: Rejection) -> list[str]:
    """The seven integrity gates of the experiment design doc, plus an eighth for R4. Empty list
    means all pass.

    Every gate is a deterministic check on the constructed Items, callable with no GPU -- this
    is the `--self-check` equivalent for the repair itself.
    """
    failures: list[str] = []
    titles = [o.title for o in item.options]

    rival_idx = next(
        (i for i, o in enumerate(item.options) if _norm(o.title) == _norm(rejection.title)), None
    )
    if rival_idx is None:
        return [f"rival {rejection.title!r} is not among this item's options"]

    r0 = conditions.get("R0")
    r1 = conditions.get("R1")
    r2 = conditions.get("R2")
    r3 = conditions.get("R3")
    r4 = conditions.get("R4")
    if r0 is None or r1 is None or r2 is None or r3 is None or r4 is None:
        return ["missing one of R0/R1/R2/R3/R4"]

    # gate 1: the named attribute was absent before the R1 edit and present after it
    before = extract.attribute_in_profile(
        item.options[rival_idx].profile, rejection.attribute, titles
    )
    if before:
        failures.append("gate1: named attribute was already present before the R1 edit")
    r1_idx, r1_appended = _edited_slot(item, r1)
    after = r1_idx == rival_idx and extract.attribute_in_profile(
        r1.options[rival_idx].profile, rejection.attribute, titles
    )
    if not after:
        failures.append("gate1: named attribute is not present in the rival's profile after R1")

    # gate 2: R2 must not accidentally repair the same attribute
    r2_idx, r2_appended = _edited_slot(item, r2)
    if r2_idx != rival_idx:
        failures.append("gate2: R2 does not edit the rival's profile")
    elif extract.attribute_in_profile(r2.options[rival_idx].profile, rejection.attribute, titles):
        failures.append("gate2: R2 accidentally repairs the named attribute")

    # gate 3: R1 and R2 inserted sentences within +-20% of each other in tokens
    if r1_appended and r2_appended:
        len1, len2 = _token_len(r1_appended[0]), _token_len(r2_appended[0])
        tolerance = 0.20 * max(len1, 1)
        if abs(len1 - len2) > tolerance + 1e-9:
            failures.append(f"gate3: R1/R2 inserted sentences differ by more than 20% in tokens "
                             f"({len1} vs {len2})")
    else:
        failures.append("gate3: could not locate both inserted sentences to compare length")

    # gate 4: date consistency -- an inserted date does not contradict an existing one
    for label, cond in (("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4)):
        idx, appended = _edited_slot(item, cond)
        if idx is None or not appended:
            continue
        entity = cond.options[idx]
        dob_sentence = find_attribute_sentence(entity, "date_of_birth")
        dod_sentence = find_attribute_sentence(entity, "date_of_death")
        birth_years, death_years = _years(dob_sentence), _years(dod_sentence)
        if birth_years and death_years and min(birth_years) >= min(death_years):
            failures.append(f"gate4: {label} leaves a date of birth on or after a date of death")

    # gate 5: the inserted sentence does not name any other candidate in the item
    for label, cond in (("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4)):
        idx, appended = _edited_slot(item, cond)
        if idx is None or not appended:
            continue
        sentence = appended[0]
        for j, other_title in enumerate(titles):
            if j == idx or not _norm(other_title):
                continue
            if _norm(other_title) in _norm(sentence):
                failures.append(f"gate5: {label} inserted sentence names {other_title!r}")

    # gate 6: the gold answer's own profile is untouched in every condition
    gold_idx = ord(item.gold_letter) - ord("A")
    for label, cond in conditions.items():
        if cond.options[gold_idx].sentences != item.options[gold_idx].sentences:
            failures.append(f"gate6: gold profile touched in {label}")

    # gate 7: option order, letters and the question are byte-identical across all five
    for label, cond in conditions.items():
        if [o.title for o in cond.options] != titles:
            failures.append(f"gate7: option order/titles differ in {label}")
        if cond.question != item.question:
            failures.append(f"gate7: question differs in {label}")

    # gate 8: R4 must edit the same option as R3 (the same target, different content) and must
    # not accidentally introduce the named attribute there -- R4's counterpart of gate 2, which
    # exists because R4's inserted sentence is irrelevant and must stay irrelevant.
    r3_idx, _r3_appended = _edited_slot(item, r3)
    r4_idx, _r4_appended = _edited_slot(item, r4)
    if r3_idx is None or r4_idx != r3_idx:
        failures.append("gate8: R4 does not edit the same option as R3")
    elif extract.attribute_in_profile(r4.options[r4_idx].profile, rejection.attribute, titles):
        failures.append("gate8: R4 accidentally repairs the named attribute")

    return failures
