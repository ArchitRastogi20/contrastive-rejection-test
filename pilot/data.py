"""Load 2WikiMultihopQA and build multiple-choice items whose options are entity profiles.

All profile text is genuine released text; nothing here writes evidence. The loader accepts
several redistributed layouts and raises with detail on anything else -- run
``run_pilot.py --inspect-schema`` before trusting a field name.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_WORD = re.compile(r"[A-Za-z0-9]+")
_TITLE_STOPWORDS = frozenset(
    "the a an of and or in on at to for de la le von van der den di del".split()
)


@dataclass
class Entity:
    title: str
    sentences: list[str]

    @property
    def profile(self) -> str:
        return " ".join(s.strip() for s in self.sentences if s and s.strip())


@dataclass
class Item:
    item_id: str
    question: str
    answer: str
    gold_title: str
    options: list[Entity]  # in presentation order
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    type_match_score: float = 0.0

    @property
    def gold_letter(self) -> str:
        for i, opt in enumerate(self.options):
            if _norm(opt.title) == _norm(self.gold_title):
                return chr(ord("A") + i)
        raise ValueError(f"gold title {self.gold_title!r} absent from options")

    def letter_of(self, title: str) -> str | None:
        for i, opt in enumerate(self.options):
            if _norm(opt.title) == _norm(title):
                return chr(ord("A") + i)
        return None


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.casefold()))


def _title_tokens(s: str) -> set[str]:
    return {t for t in _WORD.findall(s.casefold()) if t not in _TITLE_STOPWORDS}


# --------------------------------------------------------------------------- parsing


def parse_context(raw: Any) -> list[Entity]:
    """Accept the layouts 2Wiki is redistributed in; raise with detail on anything else."""
    out: list[Entity] = []

    if isinstance(raw, dict):
        # either {title: sentences} or a columnar {"title": [...], "sentences": [[...]]}
        if "title" in raw and ("sentences" in raw or "content" in raw):
            titles = raw["title"]
            bodies = raw.get("sentences", raw.get("content"))
            for t, b in zip(titles, bodies):
                out.append(Entity(str(t), _as_sentences(b)))
            return out
        for t, b in raw.items():
            out.append(Entity(str(t), _as_sentences(b)))
        return out

    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, dict):
                title = entry.get("title") or entry.get("name")
                body = entry.get("sentences", entry.get("content", entry.get("text")))
                if title is None:
                    raise ValueError(f"context entry without a title: keys={sorted(entry)}")
                out.append(Entity(str(title), _as_sentences(body)))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                out.append(Entity(str(entry[0]), _as_sentences(entry[1])))
            else:
                raise ValueError(f"unrecognised context entry of type {type(entry).__name__}")
        return out

    raise ValueError(f"unrecognised context of type {type(raw).__name__}")


def _as_sentences(body: Any) -> list[str]:
    if body is None:
        return []
    if isinstance(body, str):
        return [body]
    if isinstance(body, (list, tuple)):
        flat: list[str] = []
        for b in body:
            if isinstance(b, str):
                flat.append(b)
            elif isinstance(b, (list, tuple)):
                flat.extend(str(x) for x in b)
            else:
                flat.append(str(b))
        return flat
    return [str(body)]


def parse_evidences(raw: Any) -> list[tuple[str, str, str]]:
    """Return subject-relation-object triples. Missing or malformed evidence is not fatal."""
    triples: list[tuple[str, str, str]] = []
    if not raw:
        return triples
    for entry in raw:
        if isinstance(entry, dict):
            s = entry.get("subject") or entry.get("head")
            r = entry.get("relation") or entry.get("property")
            o = entry.get("object") or entry.get("tail")
            if s and r and o:
                triples.append((str(s), str(r), str(o)))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
            triples.append((str(entry[0]), str(entry[1]), str(entry[2])))
    return triples


# --------------------------------------------------------------------- item construction


def build_item(record: dict, n_options: int, seed: int) -> Item | None:
    """Return an Item, or None when this record cannot make a clean multiple choice.

    A record qualifies when the answer names one of the context entities and there are enough
    other entities to act as distractors. Distractors are chosen by title-token overlap with
    the gold title, which is a cheap proxy for being the same *kind* of thing -- the point of
    the design is that a rival option is a plausible answer, not an obvious throwaway.
    """
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    if not question or not answer:
        return None

    entities = [e for e in parse_context(record.get("context")) if e.profile]
    if len(entities) < n_options:
        return None

    gold = next((e for e in entities if _norm(e.title) == _norm(answer)), None)
    if gold is None:
        return None  # answers that are dates or values, not entities, are out of scope here

    gold_tokens = _title_tokens(gold.title)
    others = [e for e in entities if e is not gold]

    def overlap(e: Entity) -> tuple[int, int]:
        shared = len(gold_tokens & _title_tokens(e.title))
        return (-shared, abs(len(e.profile) - len(gold.profile)))

    others.sort(key=overlap)
    distractors = others[: n_options - 1]

    shared_counts = [len(gold_tokens & _title_tokens(e.title)) for e in distractors]
    type_match = sum(1 for c in shared_counts if c > 0) / max(1, len(distractors))

    item_id = str(record.get("_id") or record.get("id") or abs(hash(question)))
    options = [gold, *distractors]
    random.Random(f"{seed}:{item_id}").shuffle(options)

    return Item(
        item_id=item_id,
        question=question,
        answer=answer,
        gold_title=gold.title,
        options=options,
        relations=parse_evidences(record.get("evidences")),
        type_match_score=type_match,
    )


def build_items(records: Iterable[dict], n_items: int, n_options: int, seed: int) -> list[Item]:
    items: list[Item] = []
    for record in records:
        item = build_item(record, n_options=n_options, seed=seed)
        if item is not None:
            items.append(item)
        if len(items) >= n_items:
            break
    return items


# ------------------------------------------------------------------------------ loading


def load_records(dataset_id: str, split: str, limit: int) -> list[dict]:
    """Stream from the hub. Kept separate from parsing so tests never touch the network.

    Two fallbacks, both learned the hard way on this image: `datasets` 5.x dropped legacy
    script-based loading, so a repo without a parquet export needs the parquet-conversion
    revision explicitly; and a repo can simply go away, so known-good alternatives are tried
    in turn. Whichever repo answers is logged, because it belongs in the results.
    """
    import logging

    from datasets import load_dataset  # imported lazily: tests run without `datasets`

    from .config import DATASET_FALLBACKS, PARQUET_REVISION

    log = logging.getLogger(__name__)
    attempts: list[tuple[str, dict]] = []
    for repo in (dataset_id, *DATASET_FALLBACKS):
        attempts.append((repo, {}))
        attempts.append((repo, {"revision": PARQUET_REVISION}))

    last: Exception | None = None
    for repo, kwargs in attempts:
        try:
            stream = load_dataset(repo, split=split, streaming=True, **kwargs)
            out = []
            for record in stream:
                out.append(dict(record))
                if len(out) >= limit:
                    break
            if repo != dataset_id or kwargs:
                log.warning("loaded %s%s instead of %s", repo,
                            f" at {kwargs['revision']}" if kwargs else "", dataset_id)
            return out
        except Exception as exc:  # noqa: BLE001 - try the next route, report them all if none work
            log.warning("could not load %s%s: %s", repo,
                        f" at {kwargs['revision']}" if kwargs else "", exc)
            last = exc

    raise RuntimeError(f"no dataset route worked for {dataset_id}: {last}")


def load_records_from_file(path) -> list[dict]:
    """Read a local JSON or JSONL dump, for offline runs and for the fixtures."""
    text = open(path, encoding="utf-8").read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def describe_schema(records: Sequence[dict]) -> str:
    """What --inspect-schema prints. Look at this before trusting any field name above."""
    if not records:
        return "no records"
    r = records[0]
    lines = [f"{len(records)} records; first record has {len(r)} fields"]
    for k, v in r.items():
        shape = type(v).__name__
        if isinstance(v, (list, tuple)):
            shape += f"[{len(v)}]"
            if v:
                shape += f" of {type(v[0]).__name__}"
        preview = repr(v)
        if len(preview) > 160:
            preview = preview[:157] + "..."
        lines.append(f"  {k}: {shape} = {preview}")
    return "\n".join(lines)
