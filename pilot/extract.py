"""Deterministic extraction of contrastive rejections from a model's free-text explanation.

Every function is a rule over strings, not a model judgment. The attribute vocabulary is tied
to the relations 2WikiMultihopQA carries (parentage, spouse, birth and death, direction,
publication, and so on).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

from .data import Entity, Item, _norm, _title_tokens

# Attribute vocabulary: relation key -> surface cues that count as naming that attribute.
ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "date_of_birth": ("born", "birth", "birthdate", "date of birth", "b."),
    "place_of_birth": ("born in", "birthplace", "place of birth"),
    "date_of_death": ("died", "death", "date of death", "d."),
    "place_of_death": ("died in", "place of death"),
    "mother": ("mother", "maternal"),
    "father": ("father", "paternal"),
    "spouse": ("spouse", "married", "wife", "husband", "marriage"),
    "sibling": ("sibling", "brother", "sister"),
    "child": ("child", "children", "son", "daughter"),
    "director": ("director", "directed", "directing"),
    "performer": ("performer", "performed", "starring", "starred", "actor", "actress"),
    "composer": ("composer", "composed"),
    "publisher": ("publisher", "published", "publication"),
    "country": ("country", "nationality", "nation", "citizen"),
    "occupation": ("occupation", "profession", "worked as", "career"),
    "educated_at": ("educated", "education", "studied", "university", "school"),
    "employer": ("employer", "employed", "worked for"),
}

# Dropped deliberately, after the first pilot run measured what it did: a bucket matching the
# bare words "date", "year" and "when". It fired on 23 rejections and was wrong at both ends.
# A complaint that says only "no date is given" names no attribute anyone can repair -- which
# date? -- so it belongs with the vague ones. Worse, the presence check for it looked for the
# literal word "date" in the profile, which a Wikipedia sentence never contains even when it
# states one, so 16 of 17 such complaints scored as "true" by construction. Removing it raises
# the measured false-complaint rate rather than lowering it.

# A real date, for the birth/death checks below. "born" alone is not a date of birth: a profile
# reading "born in Krakow" does not answer "when was she born", and treating it as though it
# did marked true complaints false.
_MONTHS = (
    "january|february|march|april|may|june|july|august"
    "|september|october|november|december"
)

_DATE_TOKEN = re.compile(
    "(?<![A-Za-z0-9])(1[0-9]{3}|20[0-9]{2}|" + _MONTHS + ")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DATED_ATTRIBUTES = frozenset({"date_of_birth", "date_of_death"})

NEGATION_CUES: tuple[str, ...] = (
    "not ", "n't", "no ", "none", "never", "lacks", "lacking", "lacked",
    "fails to", "failed to", "absent", "missing", "without", "nothing",
    "unlike", "whereas", "rather than", "instead of", "cannot", "can not",
    "rule out", "ruled out", "ruling out", "eliminate", "eliminated",
    "exclude", "excluded", "reject", "rejected", "incorrect", "wrong",
    "does not", "do not", "did not", "is not", "are not", "was not", "were not",
)

_ABBREV = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "no.", "vs.",
    "e.g.", "i.e.", "etc.", "u.s.", "u.k.", "b.", "d.", "c.", "ca.",
)

_SENT_END = re.compile(r"(?<=[.!?])\s+")

# The letter range is derived from the item's own option count rather than hard-coded, because
# not every item has four options (Part C's items have six): a fixed A-D range makes a letter
# past D invisible, and a fixed A-F range would let a 4-option item match a spurious "E" or "F"
# that was never one of its candidates. `functools.lru_cache` means each option count compiles
# its pattern once.


@functools.lru_cache(maxsize=None)
def _letter_pick_re(n_options: int) -> re.Pattern[str]:
    last = chr(ord("A") + n_options - 1)
    return re.compile(
        rf"\b(?:answer|choice|option|candidate)\b(?:\s+(?:is|would\s+be))?"
        rf"[^A-Za-z0-9]{{0,6}}([A-{last}])\b",
        re.IGNORECASE,
    )


@functools.lru_cache(maxsize=None)
def _bare_letter_re(n_options: int) -> re.Pattern[str]:
    last = chr(ord("A") + n_options - 1)
    return re.compile(rf"(?:^|[^A-Za-z])([A-{last}])[).:\]]", re.MULTILINE)


@functools.lru_cache(maxsize=None)
def _first_line_re(n_options: int) -> re.Pattern[str]:
    last = chr(ord("A") + n_options - 1)
    return re.compile(rf"^\s*([A-{last}])[).:\]]\s*(.*)$")


# The confirmed bug: a response that opens "A) King of the Zombies" and later writes "... so I
# ruled out candidate C" was read as choosing C. It happened because the three-line head test
# right below requires the chosen option's title to be the *only* one named in the head, and a
# response that restates the rejected rival's own title in its second or third line (a common
# phrasing: "I ruled out candidate C) Some Title") makes that test see two names and give up --
# falling through to `_LETTER_PICK`, which then matches the rejection language itself.
#
# The fix checked here is the strongest possible evidence and is therefore checked first, ahead
# of the three-line test: if line one, on its own, is "X) <title>" and <title> is option X's own
# title, that is the model unambiguously declaring its choice before any rejection has even been
# written. Nothing later in the response -- rejection language included -- can produce this
# shape for the wrong letter, so it is safe to return immediately without even looking further.
def _first_line_answer(text: str, item: Item) -> str | None:
    lines = text.strip().splitlines()
    if not lines:
        return None
    m = _first_line_re(len(item.options)).match(lines[0])
    if not m:
        return None
    letter, rest = m.group(1).upper(), m.group(2)
    opt = item.options[ord(letter) - ord("A")]
    if _norm(opt.title) and _norm(opt.title) in _norm(rest):
        return letter
    return None


# How far back from a `_LETTER_PICK` match to look for a rejection/negation cue before trusting
# the match as the model's choice rather than a rival it named while ruling it out. Measured
# against the confirmed cases ("... so I ruled out candidate C.", "I ruled out candidate B) Jim
# Sterling") the cue sits 3-15 characters before the matched keyword -- "ruled out ",
# "eliminated ", "excluded " and the rest of NEGATION_CUES are all under 12 characters, plus a
# short lead-in word or two. 40 characters leaves roughly a 3x margin over the longest observed
# gap. It is also short enough that it does not read into an earlier, unrelated sentence: a
# distinct valid pick restated later in the same response ("... but the correct choice is option
# A.") sits well past 40 characters from an earlier, unrelated negation, since the two are
# separated by the connecting clause itself (empirically verified against this corpus -- see the
# validation notes for this fix).
_LEFT_CONTEXT_CHARS = 40


def _preceded_by_rejection_cue(text: str, pos: int) -> bool:
    window = text[max(0, pos - _LEFT_CONTEXT_CHARS) : pos].casefold()
    return any(cue in window for cue in NEGATION_CUES)


def split_sentences(text: str) -> list[str]:
    """Sentence split that does not break on the abbreviations this corpus is full of.

    A naive splitter corrupted a third of the units in an earlier study on adjacent data, and
    every number downstream depends on the unit, so the guard is worth twenty lines.
    """
    out: list[str] = []
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        if not line:
            continue
        buf = ""
        for piece in _SENT_END.split(line):
            buf = f"{buf} {piece}".strip() if buf else piece
            tail = buf.split()[-1].casefold() if buf.split() else ""
            if tail in _ABBREV or (len(tail) == 2 and tail.endswith(".")):
                continue  # an initial or a known abbreviation: keep accumulating
            out.append(buf)
            buf = ""
        if buf:
            out.append(buf)
    return out


# ------------------------------------------------------------------------- the choice


def parse_choice(text: str, item: Item) -> str | None:
    """Which option did the model pick? Returns a letter, or None if it cannot be read.

    Precedence, strongest evidence first:

    1. Line one is exactly "X) <X's own title>" -- the model has declared its choice before it
       has said anything else, rejection language included. See `_first_line_answer`.
    2. Title evidence in the three-line head, if it names exactly one option: a model that
       writes a name is unambiguous, while a bare letter can be part of a list.
    3. A `_LETTER_PICK` match ("the answer is B", "option C is correct") anywhere in the
       response, skipping any match whose immediate left context is a rejection/negation cue
       (see `_preceded_by_rejection_cue`) -- otherwise this rule reads "I ruled out candidate C"
       as choosing C. Kept over the whole response, not just the head, because a chosen letter
       is routinely restated in a closing line ("Therefore, the answer is B.") well past the
       first three lines, and the negation guard already removes the failure mode that scoping
       to the head would otherwise be defending against.
    4. A bare letter (`B)`, `C.`) in the head only -- riskier, so kept head-scoped as before.
    5. A single option title named anywhere in the response.

    Ties and disagreements return None rather than a guess.
    """
    first_line = _first_line_answer(text, item)
    if first_line is not None:
        return first_line

    named = [
        chr(ord("A") + i)
        for i, opt in enumerate(item.options)
        if _norm(opt.title) and _norm(opt.title) in _norm(text)
    ]
    head = "\n".join(text.strip().splitlines()[:3])
    named_head = [
        chr(ord("A") + i)
        for i, opt in enumerate(item.options)
        if _norm(opt.title) and _norm(opt.title) in _norm(head)
    ]
    if len(named_head) == 1:
        return named_head[0]

    for m in _letter_pick_re(len(item.options)).finditer(text):
        if not _preceded_by_rejection_cue(text, m.start()):
            return m.group(1).upper()
    m = _bare_letter_re(len(item.options)).search(head)
    if m:
        return m.group(1).upper()

    return named[0] if len(named) == 1 else None


# --------------------------------------------------------------------- the rejection


@dataclass
class Rejection:
    """One sentence in which the model rules a specific non-chosen option out."""

    sentence: str
    letter: str
    title: str
    matched_by: str  # "title" or "letter" or "token"
    attribute: str | None = None  # a key of ATTRIBUTES, when the defect is concrete
    cue: str | None = None
    negation_cue: str | None = None

    @property
    def is_specific(self) -> bool:
        return self.attribute is not None


@dataclass
class ItemAnalysis:
    item_id: str
    choice: str | None
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def has_rejection(self) -> bool:
        return bool(self.rejections)

    @property
    def specific(self) -> list[Rejection]:
        return [r for r in self.rejections if r.is_specific]


def _distinctive_tokens(item: Item) -> dict[int, set[str]]:
    """Tokens that identify one option and no other, so a surname can stand for a title."""
    per = [_title_tokens(o.title) for o in item.options]
    out: dict[int, set[str]] = {}
    for i, toks in enumerate(per):
        others: set[str] = set()
        for j, other in enumerate(per):
            if j != i:
                others |= other
        out[i] = {t for t in (toks - others) if len(t) >= 4}
    return out


def strip_titles(text: str, titles) -> str:
    """Blank out candidate names before looking for attribute cues.

    Measured in the first pilot run: "Robert Bres-son" supplied the cue "son" and was scored as
    a complaint about children, "When Were You Born" supplied "born", and "Charles Saunders
    (director)" supplied "director". Ten rejections were misclassified in both directions. An
    entity's name is not a claim about that entity, so it is removed before cue matching -- and
    removed from profiles too, so a birth year that appears only inside a disambiguating title
    is not counted as evidence the profile carries.
    """
    out = text
    for title in sorted((t for t in titles if t), key=len, reverse=True):
        out = re.sub(re.escape(title), " ", out, flags=re.IGNORECASE)
    return out


def find_attribute(sentence: str) -> tuple[str | None, str | None]:
    """The attribute the sentence names, if any. First match wins."""
    low = f" {sentence.casefold()} "
    for attr, cues in ATTRIBUTES.items():
        for cue in cues:
            if len(cue) <= 2:  # "b." / "d." need a word boundary or they match everything
                if re.search(rf"(?<![A-Za-z]){re.escape(cue)}", low):
                    return attr, cue
            elif cue in low:
                return attr, cue
    return None, None


def find_negation(sentence: str) -> str | None:
    low = f" {sentence.casefold()} "
    for cue in NEGATION_CUES:
        if cue in low:
            return cue.strip()
    return None


def analyse(text: str, item: Item) -> ItemAnalysis:
    """Read one model response: what it chose, and which rivals it explicitly ruled out."""
    choice = parse_choice(text, item)
    analysis = ItemAnalysis(item_id=item.item_id, choice=choice)
    distinctive = _distinctive_tokens(item)
    option_titles = [o.title for o in item.options]

    for sentence in split_sentences(text):
        neg = find_negation(sentence)
        if neg is None:
            continue
        norm_sentence = _norm(sentence)

        for i, opt in enumerate(item.options):
            letter = chr(ord("A") + i)
            if letter == choice:
                continue  # ruling out the option you picked is not a contrastive rejection

            matched_by = None
            if _norm(opt.title) and _norm(opt.title) in norm_sentence:
                matched_by = "title"
            elif distinctive[i] and distinctive[i] & set(norm_sentence.split()):
                matched_by = "token"
            elif re.search(rf"\boption {letter}\b|\bcandidate {letter}\b|\b{letter}\)", sentence):
                matched_by = "letter"
            if matched_by is None:
                continue

            attr, cue = find_attribute(strip_titles(sentence, option_titles))
            analysis.rejections.append(
                Rejection(
                    sentence=sentence.strip(),
                    letter=letter,
                    title=opt.title,
                    matched_by=matched_by,
                    attribute=attr,
                    cue=cue,
                    negation_cue=neg,
                )
            )
            break  # one rejection per sentence: the nearest-named rival owns it

    return analysis


# ------------------------------------------------- is the complaint true, and repairable


def attribute_in_profile(profile: str, attribute: str, titles=()) -> bool:
    """Does the profile actually carry this attribute?

    For a dated attribute the cue is not enough: the profile must also contain a date. "Born in
    Krakow" does not answer when someone was born, and accepting it marked true complaints false.
    """
    text = strip_titles(profile, titles)
    low = f" {text.casefold()} "
    hit = False
    for cue in ATTRIBUTES[attribute]:
        if len(cue) <= 2:
            hit = bool(re.search(rf"(?<![A-Za-z]){re.escape(cue)}", low))
        else:
            hit = cue in low
        if hit:
            break
    if not hit:
        return False
    if attribute in _DATED_ATTRIBUTES:
        return bool(_DATE_TOKEN.search(text))
    return True


def complaint_is_true(item: Item, rejection: Rejection) -> bool | None:
    """Is the named attribute really absent from the profile the model complained about?"""
    if rejection.attribute is None:
        return None
    target = next((o for o in item.options if _norm(o.title) == _norm(rejection.title)), None)
    if target is None:
        return None
    titles = [o.title for o in item.options]
    return not attribute_in_profile(target.profile, rejection.attribute, titles)


def source_of_repair(item: Item, rejection: Rejection) -> Entity | None:
    """A sibling profile in the same item that carries the missing attribute.

    Its presence is what makes the downstream repair edit *sourced* from released text rather
    than written by us, which is the whole reason this corpus was chosen.
    """
    if rejection.attribute is None:
        return None
    titles = [o.title for o in item.options]
    for opt in item.options:
        if _norm(opt.title) == _norm(rejection.title):
            continue
        if attribute_in_profile(opt.profile, rejection.attribute, titles):
            return opt
    return None


def is_usable(item: Item, rejection: Rejection) -> bool:
    """The pilot's unit of account: specific, accurate, and repairable from released text."""
    return (
        rejection.is_specific
        and complaint_is_true(item, rejection) is True
        and source_of_repair(item, rejection) is not None
    )
