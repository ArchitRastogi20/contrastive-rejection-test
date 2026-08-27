"""E3: necessity -- if the model chose X and justified it by saying rival Y lacks attribute A,
is A load-bearing for X itself?

The project's existing experiment (`pilot.run_experiment`) tests sufficiency: a model rejects a
rival by naming a missing fact, the experimenter inserts that fact into the rival's profile, and
the choice is re-measured. Its own report states the gap this module closes: "R1-R2 and R3-R4
test sufficiency, not necessity, since nothing here removes the fact from a profile already
lacking it." E3 is the mirror -- remove the sentence stating A from X's *own* profile instead,
and see whether the choice moves away from X. If it does, A was doing work; if it does not, "Y
lacks A" was decoration that happened to be true.

Three conditions per qualifying item, N0-N2, in the same shape `pilot.repair`'s R0-R4 already
use: N0 is the unedited item (an integrity check under greedy decoding, not a treatment); N1
deletes the chosen option's own sentence stating A; N2 deletes a different, length-matched
sentence from the same profile instead, so a flip under N1 is not merely "any deletion confuses
the model." Population and conditions are built entirely from committed stage-1 records
(`pilot.extract.analyse`/`is_usable`, reused unchanged) and the item's own already-released
profile text -- no new stage-1 generation, no repair search, no sibling sourcing: unlike R1/R2,
N1/N2 never touch any profile but the chosen option's own.

    python -m pilot.run_necessity --self-check              # gate logic on the fixture, no GPU
    python -m pilot.run_necessity --dry-run                 # whole pipeline, stub model, no GPU
    python -m pilot.run_necessity --part A --models Qwen2.5-7B   # the real thing

Every field is read off the model's own text by `pilot.extract`; no LLM judges anything here.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import config as C
from . import extract, prompts, repair
from .data import Entity, Item, _norm, _WORD, build_items, load_records, load_records_from_file
from .extract import Rejection
from .models import Backend, ModelUnavailable, StubBackend, append_letter_probe, get_backend
from .run_experiment import mcnemar
from .run_pilot import _StubReplies, _slug
from .watchdog import Watchdog, commit_gpu_seconds, cumulative_gpu_seconds

log = logging.getLogger("run_necessity")

CONDITION_LABELS = ("N0", "N1", "N2")

# Run 3's stage 3 cost about 11.9 GPU-seconds per item across five conditions, each
# condition one free-text generation plus one forced-letter probe, so about 2.4s per
# condition on a 7-8B model at four options. E3 runs three conditions, so about 7.2s an
# item. That is a measured figure from `code/results/gpu_ledger.csv` rather than a guess,
# but it is a 7-8B four-option figure: re-derive it from the first real run's ledger row
# before trusting it for a larger checkpoint.
EST_SECONDS_PER_CONDITION = 2.4

# Mirrors `pilot.surprisal.PARTS`/`pilot.probe_variants.PARTS` exactly (each of those modules
# keeps its own copy rather than sharing one -- see their docstrings). Run 3's committed
# stage-1/2/3 output is split across these three directories (A, B at 4 options; C at 6).
PARTS = {"A": "exp3a", "B": "exp3b", "C": "exp3c"}

_N_ITEMS_GROWTH_CAP = 4000


class NecessityUnavailable(Exception):
    """N0/N1/N2 cannot be mechanically constructed for this item -- raised, never worked
    around: the caller counts the drop and records this message as the reason, the same
    discipline `repair.RepairUnavailable` uses for R0-R4.

    `gate_failures` carries the integrity gates' own failures, when that is why the build was
    refused, so the caller can tally which gate is doing the excluding rather than only that
    the item was dropped.
    """

    def __init__(self, message: str, *, gate_failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.gate_failures = gate_failures


@dataclass
class SelectedItem:
    """One item that qualified for E3, with the N0/N1/N2 conditions already built and gated."""

    item: Item
    rejection: Rejection
    chosen_letter: str
    conditions: dict[str, Item]
    chosen_is_gold: bool
    question_relevant_attribute: bool


# --------------------------------------------------------------------------- building N0/N1/N2


def _token_len(text: str) -> int:
    return len(_WORD.findall(text))


def find_chosen_attribute_sentence(entity: Entity, attribute: str) -> tuple[int, str] | None:
    """The index and text of the one sentence in `entity.sentences` that states `attribute`, or
    None.

    Reuses `repair.find_attribute_sentence` (itself the extractor's own cue matching plus the
    dated-attribute date-token rule) to decide *whether* a sentence states the attribute, so the
    answer here agrees exactly with `attribute_in_profile`'s own reading -- never a second,
    independent rule. That function reads off the joined, re-split profile text; deleting a
    sentence needs the matching entry in the raw `entity.sentences` list instead, so the
    sentence text it returns is located there by exact (stripped) match. Returns None -- and the
    item is dropped, not guessed at -- on the rare case a profile entry spans more than one of
    `split_sentences`'s units, so no single list entry corresponds to the attribute sentence
    cleanly.
    """
    sentence = repair.find_attribute_sentence(entity, attribute)
    if sentence is None:
        return None
    for i, raw in enumerate(entity.sentences):
        if raw.strip() == sentence.strip():
            return i, raw
    return None


def _select_n2_index(
    entity: Entity, n1_index: int, attribute: str, target_len: int
) -> int | None:
    """The control deletion for N2: a different sentence in `entity.sentences` -- one that does
    not itself state `attribute` -- closest to `target_len` in tokens, first-in-order tie-break.
    The same "closest length, then existing order" preference `repair._r2_candidates_ordered`
    uses for R2's source, applied here to the chosen profile's own other sentences rather than
    to a sibling's.
    """
    candidates: list[tuple[int, int]] = []
    for i, raw in enumerate(entity.sentences):
        if i == n1_index:
            continue
        stripped = extract.strip_titles(raw, [entity.title])
        attr, _cue = extract.find_attribute(stripped)
        states_attribute = attr == attribute and not (
            attribute in extract._DATED_ATTRIBUTES and not extract._DATE_TOKEN.search(raw)
        )
        if states_attribute:
            continue
        candidates.append((abs(_token_len(raw) - target_len), i))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def build_necessity_conditions(
    item: Item, chosen_letter: str, attribute: str
) -> tuple[dict[str, Item], dict]:
    """N0 (unedited), N1 (the chosen profile's sentence stating `attribute` deleted), N2 (a
    different, length-matched sentence deleted instead) -- or raise NecessityUnavailable.

    A deterministic, pure transform on strings and dataclasses, mirroring `repair.
    build_conditions`'s discipline: given the same item, chosen letter and attribute, every call
    returns byte-identical output. Every integrity gate (`check_integrity_necessity`) is checked
    before returning, exactly as `repair.build_conditions_with_diagnostics` checks its own gates
    before yielding R0-R4 -- a caller never receives conditions that have not already passed
    every gate.
    """
    chosen_idx = ord(chosen_letter) - ord("A")
    if chosen_idx < 0 or chosen_idx >= len(item.options):
        raise NecessityUnavailable(
            f"chosen letter {chosen_letter!r} is not among this item's options"
        )
    chosen = item.options[chosen_idx]

    found = find_chosen_attribute_sentence(chosen, attribute)
    if found is None:
        raise NecessityUnavailable(
            f"named attribute {attribute!r} is not stated in one isolable sentence of the "
            "chosen option's own profile"
        )
    n1_index, n1_sentence = found
    if len(chosen.sentences) < 2:
        raise NecessityUnavailable(
            "chosen profile has fewer than two sentences; no control deletion exists"
        )

    target_len = _token_len(n1_sentence)
    n2_index = _select_n2_index(chosen, n1_index, attribute, target_len)
    if n2_index is None:
        raise NecessityUnavailable(
            "no length-matched control sentence available in the chosen profile"
        )
    n2_sentence = chosen.sentences[n2_index]

    n0 = copy.deepcopy(item)
    n1 = copy.deepcopy(item)
    del n1.options[chosen_idx].sentences[n1_index]
    n2 = copy.deepcopy(item)
    del n2.options[chosen_idx].sentences[n2_index]
    conditions = {"N0": n0, "N1": n1, "N2": n2}

    failures = check_integrity_necessity(conditions, item, chosen_idx, attribute)
    if failures:
        raise NecessityUnavailable(
            f"conditions failed the integrity gates: {'; '.join(failures)}",
            gate_failures=failures,
        )

    diagnostics = {
        "chosen_letter": chosen_letter, "n1_index": n1_index, "n1_sentence": n1_sentence,
        "n2_index": n2_index, "n2_sentence": n2_sentence,
    }
    return conditions, diagnostics


# ----------------------------------------------------------------------------- integrity gates


def _removed_sentence(base: Item, cond: Item, idx: int) -> str | None:
    """The one sentence `cond.options[idx].sentences` is missing relative to
    `base.options[idx].sentences`, found by diffing the two lists rather than trusted from the
    construction call -- the same independence `repair.check_integrity`'s `_edited_slot` keeps
    from `_with_sentence_appended`. None when the lists are not related by exactly one removal
    with everything else in the same order.
    """
    base_list = base.options[idx].sentences
    cond_list = cond.options[idx].sentences
    if len(cond_list) != len(base_list) - 1:
        return None
    i = 0
    for s in base_list:
        if i < len(cond_list) and s == cond_list[i]:
            i += 1
        else:
            return s
    return None


def check_integrity_necessity(
    conditions: dict[str, Item], item: Item, chosen_idx: int, attribute: str
) -> list[str]:
    """The seven necessity gates. Empty list means all pass.

    Mirrors `repair.check_integrity`'s structure and its discipline of independent, diff-based
    verification: every check re-derives what changed by comparing the built Items against the
    untouched `item`, never by trusting the variables `build_necessity_conditions` used to build
    them. Every gate is a deterministic check on the constructed Items, callable with no GPU --
    this is the `--self-check` equivalent for the necessity build itself.
    """
    failures: list[str] = []
    titles = [o.title for o in item.options]

    n0, n1, n2 = conditions.get("N0"), conditions.get("N1"), conditions.get("N2")
    if n0 is None or n1 is None or n2 is None:
        return ["missing one of N0/N1/N2"]

    # gate 1: A present in the chosen profile before N1, absent after
    before = extract.attribute_in_profile(item.options[chosen_idx].profile, attribute, titles)
    if not before:
        failures.append("gate1: named attribute is not present in the chosen profile before N1")
    after = extract.attribute_in_profile(n1.options[chosen_idx].profile, attribute, titles)
    if after:
        failures.append("gate1: named attribute is still present in the chosen profile after N1")

    # gate 2: N2 removes a sentence and leaves A still present
    n2_removed = _removed_sentence(item, n2, chosen_idx)
    if n2_removed is None:
        failures.append("gate2: N2 does not remove exactly one sentence from the chosen profile")
    elif not extract.attribute_in_profile(n2.options[chosen_idx].profile, attribute, titles):
        failures.append("gate2: N2 accidentally removes the named attribute")

    # gate 3: N1's and N2's deleted sentences are within 20% of each other in tokens
    n1_removed = _removed_sentence(item, n1, chosen_idx)
    if n1_removed is None:
        failures.append("gate3: N1 does not remove exactly one sentence from the chosen profile")
    elif n2_removed is not None:
        len1, len2 = _token_len(n1_removed), _token_len(n2_removed)
        tolerance = 0.20 * max(len1, 1)
        if abs(len1 - len2) > tolerance + 1e-9:
            failures.append(
                f"gate3: N1/N2 deleted sentences differ by more than 20% in tokens "
                f"({len1} vs {len2})"
            )
    else:
        failures.append("gate3: could not locate both deleted sentences to compare length")

    # gate 4: both conditions leave the chosen profile with at least one sentence
    if len(n1.options[chosen_idx].sentences) < 1:
        failures.append("gate4: N1 leaves the chosen profile with no sentences")
    if len(n2.options[chosen_idx].sentences) < 1:
        failures.append("gate4: N2 leaves the chosen profile with no sentences")

    # gate 5: the deleted sentence names no other candidate in the item
    for label, removed in (("N1", n1_removed), ("N2", n2_removed)):
        if removed is None:
            continue
        for j, other_title in enumerate(titles):
            if j == chosen_idx or not _norm(other_title):
                continue
            if _norm(other_title) in _norm(removed):
                failures.append(f"gate5: {label} deleted sentence names {other_title!r}")

    # gate 6: every profile other than the chosen option's is unchanged in every condition
    for label, cond in conditions.items():
        for j in range(len(item.options)):
            if j == chosen_idx:
                continue
            if cond.options[j].sentences != item.options[j].sentences:
                failures.append(
                    f"gate6: non-chosen profile {item.options[j].title!r} touched in {label}"
                )

    # gate 7: option order, titles and question are byte-identical across N0/N1/N2
    for label, cond in conditions.items():
        if [o.title for o in cond.options] != titles:
            failures.append(f"gate7: option order/titles differ in {label}")
        if cond.question != item.question:
            failures.append(f"gate7: question differs in {label}")

    return failures


# --------------------------------------------------------------------- question-relevance flag

# ponytail: a keyword scan, not a parse of the question's real logical structure. Ceiling: it
# will not catch every comparison phrasing (e.g. a nationality/location comparison naming no
# order word), and it can fire on a question that uses one of these words for an unrelated
# reason. It is deliberately conservative in what it is used for -- a stratification flag, never
# a gate -- so a false negative here costs precision in a downstream split, not a wrong drop.
# Upgrade path: once real E3 items are open, read a sample of question text (rule 5) and tighten
# or extend this list against what is actually there.
_ORDER_CUES = ("first", "earlier", "earliest", "later", "latest", "before", "after",
               "younger", "older")


def is_question_relevant_attribute(question: str, attribute: str) -> bool:
    """Does this look like a date/order comparison receiving a dated attribute -- the case the
    task instructions call out to record for stratification rather than silently mix into the
    headline result. Removing A can change which option is actually correct precisely in this
    case, so it is recorded, never gated on.
    """
    if attribute not in extract._DATED_ATTRIBUTES:
        return False
    low = question.casefold()
    return any(cue in low for cue in _ORDER_CUES)


# --------------------------------------------------------- reading committed stage-1 records


def _rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _stage1_rows_by_model(root: Path, *, recursive: bool) -> dict[str, list[dict]]:
    """Every committed stage-1 row under `root`, grouped by model. Same convention `pilot.
    surprisal._stage1_rows_by_model` uses -- kept as its own copy here, this project's
    established per-experiment-script pattern (`run_experiment.py`, `surprisal.py`,
    `probe_variants.py` each keep a thin copy of this kind of helper rather than sharing one).
    """
    pattern = "**/stage1_*.jsonl" if recursive else "stage1_*.jsonl"
    out: dict[str, list[dict]] = {}
    for path in sorted(root.glob(pattern)):
        for row in _rows(path):
            out.setdefault(row["model"], []).append(row)
    return out


def _models_in_part(exp_dir: Path) -> list[str]:
    """Which models this part actually ran, read off whichever stage file the directory has:
    stage2, then stage3, then stage1. A part can replay stage 1 from elsewhere (exp3a carries no
    stage1_*.jsonl of its own -- see `surprisal._stage1_rows_by_model`'s docstring), so its own
    stage1 files cannot be trusted to name every model; its stage2/stage3 files always can.
    """
    for pattern in ("stage2_*.jsonl", "stage3_*.jsonl", "stage1_*.jsonl"):
        models = {row["model"] for path in sorted(exp_dir.glob(pattern)) for row in _rows(path)}
        if models:
            return sorted(models)
    return []


def _rebuild_item_set(
    *, n_items: int, n_options: int, scan: int, data_file: Path | None
) -> list[Item]:
    records = (
        load_records_from_file(data_file) if data_file
        else load_records(C.DATASET_ID, C.DATASET_SPLIT, scan)
    )
    return build_items(records, n_items=n_items, n_options=n_options, seed=C.SEED)


def _rejection_from_stage1_row(item: Item, row: dict) -> Rejection | None:
    """Recompute the usable `Rejection` the row's own `response` selected -- one code path, the
    extractor's own, same discipline `run_experiment._selected_rejection`/`surprisal.
    _rejection_from_stage1_row` use.
    """
    if not row.get("selected"):
        return None
    analysis = extract.analyse(row["response"], item)
    return next((r for r in analysis.rejections if extract.is_usable(item, r)), None)


_DROP_CATEGORY = (
    ("is not among this item's options", "chosen_letter_invalid"),
    ("is not stated in one isolable sentence", "attribute_not_in_chosen_profile"),
    ("fewer than two sentences", "insufficient_sentences"),
    ("no length-matched control sentence", "no_n2_candidate"),
    ("failed the integrity gates", "integrity_gate_failed"),
)


def _drop_category(reason: str) -> str:
    for needle, category in _DROP_CATEGORY:
        if needle in reason:
            return category
    return "other_necessity_unavailable"


def estimate_gpu_seconds(n_items: int) -> float:
    """`n_items` x the three conditions x `EST_SECONDS_PER_CONDITION`."""
    return n_items * len(CONDITION_LABELS) * EST_SECONDS_PER_CONDITION


def budget_check(
    n_items: int, *, spent: float, allowance: float = C.PROJECT_GPU_BUDGET_S,
) -> dict:
    """Same shape and the same purity guarantee as `pilot.probe_variants.budget_check`: no
    ledger I/O, so it is testable with a hand-supplied `spent`."""
    estimate = estimate_gpu_seconds(n_items)
    remaining = allowance - spent
    return {
        "n_items": n_items, "estimate_s": estimate, "spent_s": spent,
        "allowance_s": allowance, "remaining_s": remaining, "ok": estimate <= remaining,
    }


def select_items(
    *, part: str, results_dir: Path = C.RESULTS_DIR, n_options: int | None = None,
    scan: int | None = None, data_file: Path | None = None, models: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, SelectedItem]], dict]:
    """{model: {item_id: SelectedItem}} for every committed stage-1 row that qualifies for E3,
    plus the funnel/drop counts.

    Deliberately lighter than `surprisal.reconstruct_conditions`: E3 needs only the item's own
    profiles and the model's own committed stage-1 response, never a corpus index or a repair
    search across siblings (N1/N2 only ever touch the chosen option's own profile), so this
    reads committed stage-1 records directly rather than depending on stage-2's repair having
    succeeded -- the population here is a property of stage 1 alone, per the task design.
    """
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; expected one of {sorted(PARTS)}")
    exp_dir = results_dir / PARTS[part]
    summary_path = exp_dir / "experiment_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    )
    resolved_n_options = n_options or summary.get("n_options") or C.N_OPTIONS

    stage1_local = _stage1_rows_by_model(exp_dir, recursive=False)
    stage1_global: dict[str, list[dict]] | None = None
    model_names = _models_in_part(exp_dir)

    counts: dict = {
        "part": part, "exp_dir": str(exp_dir), "n_options": resolved_n_options, "models": {},
    }
    out: dict[str, dict[str, SelectedItem]] = {}

    items_list: list[Item] = []
    items_by_id: dict[str, Item] = {}

    def _grow_to(min_n: int) -> None:
        nonlocal items_list, items_by_id
        if len(items_list) >= min_n:
            return
        items_list = _rebuild_item_set(
            n_items=min_n, n_options=resolved_n_options, scan=min_n * 40, data_file=data_file,
        )
        items_by_id = {i.item_id: i for i in items_list}

    for model in model_names:
        if models and not any(m in model for m in models):
            continue
        model_counts = {
            "attempted": 0, "unreadable_choice": 0, "missing_item": 0,
            "option_order_mismatch": 0, "no_usable_rejection": 0, "kept": 0,
            "drop_reasons": {}, "gate_failures": {},
        }

        stage1_rows_for_model = stage1_local.get(model)
        if not stage1_rows_for_model:
            if stage1_global is None:
                stage1_global = _stage1_rows_by_model(results_dir, recursive=True)
            stage1_rows_for_model = stage1_global.get(model)
        if not stage1_rows_for_model:
            out[model] = {}
            counts["models"][model] = model_counts
            continue

        # A row from a different n_options build has an option_titles of the wrong length and
        # must never be trusted here -- same filter `surprisal.reconstruct_conditions` applies.
        stage1_rows_for_model = [
            r for r in stage1_rows_for_model
            if len(r.get("option_titles") or []) == resolved_n_options
        ]
        if not stage1_rows_for_model:
            out[model] = {}
            counts["models"][model] = model_counts
            continue

        wanted_ids = {r["item_id"] for r in stage1_rows_for_model}
        _grow_to(len(stage1_rows_for_model))
        n = len(items_list)
        while not (wanted_ids <= set(items_by_id)) and n < _N_ITEMS_GROWTH_CAP:
            n = min(_N_ITEMS_GROWTH_CAP, max(n * 2, n + 1))
            _grow_to(n)

        kept: dict[str, SelectedItem] = {}
        for row in stage1_rows_for_model:
            model_counts["attempted"] += 1
            if row.get("choice") is None:
                model_counts["unreadable_choice"] += 1
                continue
            item = items_by_id.get(row["item_id"])
            if item is None:
                model_counts["missing_item"] += 1
                continue
            if row.get("option_titles") != [o.title for o in item.options]:
                model_counts["option_order_mismatch"] += 1
                continue

            rejection = _rejection_from_stage1_row(item, row)
            if rejection is None:
                model_counts["no_usable_rejection"] += 1
                continue

            try:
                conditions, _diag = build_necessity_conditions(
                    item, row["choice"], rejection.attribute
                )
            except NecessityUnavailable as exc:
                category = _drop_category(str(exc))
                model_counts["drop_reasons"][category] = (
                    model_counts["drop_reasons"].get(category, 0) + 1
                )
                for f in exc.gate_failures or ():
                    gate = f.split(":", 1)[0]
                    model_counts["gate_failures"][gate] = (
                        model_counts["gate_failures"].get(gate, 0) + 1
                    )
                continue

            kept[row["item_id"]] = SelectedItem(
                item=item, rejection=rejection, chosen_letter=row["choice"],
                conditions=conditions, chosen_is_gold=row["choice"] == item.gold_letter,
                question_relevant_attribute=is_question_relevant_attribute(
                    item.question, rejection.attribute
                ),
            )
            model_counts["kept"] += 1

        out[model] = kept
        counts["models"][model] = model_counts

    return out, counts


# ------------------------------------------------------------------------------- the run itself


def run_stage_necessity(
    backend: Backend, kept: dict[str, SelectedItem], out_dir: Path, part: str | None,
) -> tuple[dict[str, dict[str, bool | None]], dict]:
    """Re-ask under N0/N1/N2 for every gate-clean item. Two backend calls per condition: the
    existing free-text answer-and-explanation call, plus the forced-choice letter-probe read on
    the item's own originally-chosen letter (`models.append_letter_probe`, reused unchanged).
    Writes stage_necessity_<model>.jsonl and returns per-item `still_chooses_original` by
    condition (for the paired N1-vs-N2 and N1-vs-N0 contrasts) plus the funnel/condition counts.
    """
    label = f"{_slug(backend.name)}/necessity"
    path = out_dir / f"stage_necessity_{_slug(backend.name)}.jsonl"
    total = len(kept) * len(CONDITION_LABELS) * 2  # free-text call + letter-probe call, each
    dog = Watchdog(label=label, total=total, budget_min=C.PER_MODEL_BUDGET_MIN,
                    heartbeat=out_dir / "heartbeat.json")
    dog.sample_vram()

    outcomes: dict[str, dict[str, bool | None]] = {}
    condition_counts = {
        c: {"still_chooses_original": 0, "unreadable": 0, "n": 0} for c in CONDITION_LABELS
    }
    n0_reproduced = 0
    n0_flipped = 0
    abort_reason: str | None = None

    try:
        with open(path, "a", encoding="utf-8") as fh:
            for item_id, sel in kept.items():
                if dog.should_abort:
                    abort_reason = dog.abort_reason
                    break
                original_letter = sel.chosen_letter

                per_item: dict[str, bool | None] = {}
                r0_probs: dict[str, float] = {}
                r0_complete = False

                for label_c in CONDITION_LABELS:
                    if dog.should_abort:
                        abort_reason = dog.abort_reason
                        break
                    cond_item = sel.conditions[label_c]
                    chat = prompts.render(cond_item, "elicited")
                    response = backend.generate([chat])[0]
                    analysis = extract.analyse(response, cond_item)
                    still_original = (
                        None if analysis.choice is None else analysis.choice == original_letter
                    )
                    per_item[label_c] = still_original

                    condition_counts[label_c]["n"] += 1
                    if analysis.choice is None:
                        condition_counts[label_c]["unreadable"] += 1
                    elif still_original:
                        condition_counts[label_c]["still_chooses_original"] += 1
                    if label_c == "N0":
                        if still_original is True:
                            n0_reproduced += 1
                        elif still_original is False:
                            n0_flipped += 1

                    # The second, additional call: same rendered context, one appended
                    # instruction, forced to a single token -- never touches anything above.
                    candidates = [chr(ord("A") + i) for i in range(len(cond_item.options))]
                    probe_chat = append_letter_probe(chat)
                    letter_read = backend.letter_probs([probe_chat], [candidates])[0]
                    p_original = letter_read.probs.get(original_letter)
                    if label_c == "N0":
                        r0_probs, r0_complete = letter_read.probs, letter_read.complete

                    r0_p_target = r0_probs.get(original_letter) if r0_complete else None
                    delta_p_original = None
                    if (
                        letter_read.complete and r0_complete
                        and r0_p_target is not None and p_original is not None
                    ):
                        delta_p_original = p_original - r0_p_target

                    fh.write(json.dumps({
                        "model": backend.name, "part": part, "item_id": item_id,
                        "condition": label_c, "attribute": sel.rejection.attribute,
                        "rival_letter": sel.rejection.letter, "rival_title": sel.rejection.title,
                        "chosen_letter": original_letter, "chosen_is_gold": sel.chosen_is_gold,
                        "question_relevant_attribute": sel.question_relevant_attribute,
                        "choice": analysis.choice, "still_chooses_original": still_original,
                        "letter_probe": letter_read.to_dict(), "p_original": p_original,
                        "r0_p_target": r0_p_target, "delta_p_original": delta_p_original,
                        "response": response,  # always kept
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    dog.tick(2)
                outcomes[item_id] = per_item
                if abort_reason:
                    break
    finally:
        pass

    if backend.kind != "stub":
        commit_gpu_seconds(label=label, model=backend.name, backend=backend.kind,
                            items=len(outcomes), seconds=dog.elapsed_s,
                            vram_peak=dog.vram_peak, abort_reason=abort_reason)

    counts = {
        "n_items": len(outcomes), "condition_counts": condition_counts,
        # N0 is the integrity check, not a treatment: a flip here means nothing else about this
        # item is interpretable (see the module docstring), so it is counted and excluded from
        # the paired contrasts below, never silently pooled in with genuine N1/N2 flips.
        "n0_reproduced": n0_reproduced, "n0_flipped": n0_flipped,
        "abort_reason": abort_reason, "elapsed_min": round(dog.elapsed_s / 60.0, 2),
        "vram_peak_pct": round(dog.vram_peak * 100, 1),
    }
    return outcomes, counts


def _interpretable(outcomes: dict[str, dict[str, bool | None]]) -> dict[str, dict[str, bool | None]]:
    """Items whose N0 (unedited) re-ask reproduced the original stage-1 choice -- see
    `run_stage_necessity`'s note on why a flip there makes the item uninterpretable."""
    return {item_id: per for item_id, per in outcomes.items() if per.get("N0") is True}


def necessity_contrasts(outcomes: dict[str, dict[str, bool | None]]) -> dict:
    """The paired contrasts the task calls for: N1 vs N2 (primary), N1 vs N0 (secondary), both
    restricted to N0-reproducing items and reusing `run_experiment.mcnemar` unchanged so the
    paired discordant counts (b, c) and the exact two-sided p are legible without extra tooling.
    """
    interpretable = _interpretable(outcomes)
    return {
        "n_items_total": len(outcomes), "n_items_interpretable": len(interpretable),
        "n_items_excluded_n0_uninterpretable": len(outcomes) - len(interpretable),
        "mcnemar_n1_vs_n2": mcnemar(interpretable, "N1", "N2"),
        "mcnemar_n1_vs_n0": mcnemar(interpretable, "N1", "N0"),
    }


# --------------------------------------------------------------------------- self-check


def _fixture_item_for_gates() -> tuple[Item, str, str]:
    """An item, the model's stage-1 choice letter, and the attribute its rejection named --
    built so `build_necessity_conditions` succeeds cleanly on it. Every gate-violation test below
    starts from this clean build and then deliberately breaks one gate at a time, mirroring
    `run_experiment._fixture_item_for_gates`/`_gate_violation_problems`.
    """
    item = Item(
        item_id="necessity-self-check-1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", [
                "Anna Kowalska is a Polish filmmaker.",
                "She was born in 1970 in Krakow.",
                "She directed Blue River.",
            ]),
            Entity("Bruno Kowalski", [
                "Bruno Kowalski is a Polish cinematographer.",
                "He worked on several feature films.",
            ]),
            Entity("Clara Novak", [
                "Clara Novak is a Czech screenwriter.",
                "Her mother was named Maria Elena Novak.",
            ]),
            Entity("Dawid Kowalski", [
                "Dawid Kowalski is a Polish film editor.",
                "He edited three documentaries.",
            ]),
        ],
    )
    return item, "A", "date_of_birth"


def _gate_violation_problems() -> list[str]:
    """Deliberately break each of the seven gates and confirm `check_integrity_necessity` catches
    it. Independent of `test_necessity.py`: this runs from the command line with no pytest and no
    GPU, so a regression in the gates cannot pass silently just because a particular run's data
    never happened to exercise one of them.
    """
    item, chosen_letter, attribute = _fixture_item_for_gates()
    chosen_idx = ord(chosen_letter) - ord("A")
    conditions, _diag = build_necessity_conditions(item, chosen_letter, attribute)
    baseline_failures = check_integrity_necessity(conditions, item, chosen_idx, attribute)
    problems: list[str] = []
    if baseline_failures:
        problems.append(f"a clean build already fails: {baseline_failures}")
        return problems

    def gate_fires(label: str, broken: dict, prefix: str) -> str | None:
        failures = check_integrity_necessity(broken, item, chosen_idx, attribute)
        if not any(f.startswith(prefix) for f in failures):
            return f"{label}: expected a failure starting with {prefix!r}, got {failures}"
        return None

    # gate 1: put the deleted (date of birth) sentence back into N1
    broken = copy.deepcopy(conditions)
    broken["N1"].options[0].sentences.insert(1, "She was born in 1970 in Krakow.")
    if (p := gate_fires("gate1", broken, "gate1:")) is not None:
        problems.append(p)

    # gate 2: N2 removes the date-of-birth sentence instead of the length-matched control
    broken = copy.deepcopy(conditions)
    broken["N2"].options[0].sentences = [
        "Anna Kowalska is a Polish filmmaker.", "She directed Blue River.",
    ]
    if (p := gate_fires("gate2", broken, "gate2:")) is not None:
        problems.append(p)

    # gate 3: N2 instead removes a far-shorter sentence than N1's
    broken = copy.deepcopy(conditions)
    broken["N2"].options[0].sentences = [
        "Anna Kowalska is a Polish filmmaker.", "She was born in 1970 in Krakow.",
    ]
    if (p := gate_fires("gate3", broken, "gate3:")) is not None:
        problems.append(p)

    # gate 4: delete every sentence from N1's chosen profile
    broken = copy.deepcopy(conditions)
    broken["N1"].options[0].sentences = []
    if (p := gate_fires("gate4", broken, "gate4:")) is not None:
        problems.append(p)

    # gate 5: the deleted sentence names another candidate. A separate base item is needed here
    # (rather than the shared `gate_fires`/`item` closure above) since this scenario's own N1/N2
    # are diffed against a differently-shaped chosen profile.
    item5 = copy.deepcopy(item)
    item5.options[0].sentences = [
        "Anna Kowalska is a Polish filmmaker.",
        "She was born in 1970 in Krakow.",
        "She once worked alongside Bruno Kowalski on a project.",
        "She directed Blue River.",
    ]
    conditions5, _diag5 = build_necessity_conditions(item5, "A", "date_of_birth")
    broken = copy.deepcopy(conditions5)
    broken["N2"] = copy.deepcopy(item5)
    broken["N2"].options[0].sentences = [
        s for s in item5.options[0].sentences if "Bruno Kowalski" not in s
    ]
    failures5 = check_integrity_necessity(broken, item5, chosen_idx, attribute)
    if not any(f.startswith("gate5:") for f in failures5):
        problems.append(f"gate5: expected a failure starting with 'gate5:', got {failures5}")

    # gate 6: touch a non-chosen profile in N1
    broken = copy.deepcopy(conditions)
    broken["N1"].options[1].sentences.append("This sentence should never be here.")
    if (p := gate_fires("gate6", broken, "gate6:")) is not None:
        problems.append(p)

    # gate 7: reorder the options in one condition
    broken = copy.deepcopy(conditions)
    broken["N1"].options[0], broken["N1"].options[1] = (
        broken["N1"].options[1], broken["N1"].options[0]
    )
    if (p := gate_fires("gate7", broken, "gate7:")) is not None:
        problems.append(p)

    return problems


def self_check() -> int:
    """Item/condition construction on the fixture, plus all seven integrity gates -- no GPU, no
    network. Exits non-zero on any failure, the equivalent of `run_experiment.self_check` for E3.
    """
    C.setup_logging()
    records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
    items = build_items(records, n_items=len(records), n_options=C.N_OPTIONS, seed=C.SEED)
    if not items:
        log.error("self-check: no items built from the fixture")
        return 2

    backend = StubBackend(_StubReplies(), name="self-check-stub")

    problems: list[str] = []
    built, dropped = 0, 0
    for item in items:
        response = backend.generate([prompts.render(item, "elicited")])[0]
        analysis = extract.analyse(response, item)
        if analysis.choice is None:
            continue
        rejection = next((r for r in analysis.rejections if extract.is_usable(item, r)), None)
        if rejection is None:
            continue
        try:
            conditions, _diag = build_necessity_conditions(
                item, analysis.choice, rejection.attribute
            )
        except NecessityUnavailable:
            dropped += 1
            continue
        chosen_idx = ord(analysis.choice) - ord("A")
        failures = check_integrity_necessity(conditions, item, chosen_idx, rejection.attribute)
        if failures:
            problems.append(f"{item.item_id}: gates failed after a clean build: {failures}")
        else:
            built += 1

    problems += _gate_violation_problems()

    log.info("self-check: %d item(s) built cleanly, %d dropped as unbuildable, %d problem(s)",
              built, dropped, len(problems))
    for p in problems:
        log.error("self-check FAILED: %s", p)
    if not problems:
        log.info("self-check PASSED")
    return 0 if not problems else 1


# --------------------------------------------------------------------------- dry run, no GPU


def _dry_run_reconstruction() -> dict[str, SelectedItem]:
    """The same construction `self_check` exercises on the fixture, packaged as `SelectedItem`s
    so `--dry-run` runs `run_stage_necessity` -- the exact code path a real part would use --
    without touching `code/results/` or the network at all."""
    records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
    items = build_items(records, n_items=len(records), n_options=C.N_OPTIONS, seed=C.SEED)
    backend = StubBackend(_StubReplies(), name="stub")

    kept: dict[str, SelectedItem] = {}
    for item in items:
        response = backend.generate([prompts.render(item, "elicited")])[0]
        analysis = extract.analyse(response, item)
        if analysis.choice is None:
            continue
        rejection = next((r for r in analysis.rejections if extract.is_usable(item, r)), None)
        if rejection is None:
            continue
        try:
            conditions, _diag = build_necessity_conditions(
                item, analysis.choice, rejection.attribute
            )
        except NecessityUnavailable:
            continue
        kept[item.item_id] = SelectedItem(
            item=item, rejection=rejection, chosen_letter=analysis.choice, conditions=conditions,
            chosen_is_gold=analysis.choice == item.gold_letter,
            question_relevant_attribute=is_question_relevant_attribute(
                item.question, rejection.attribute
            ),
        )
    return kept


def _run_dry(args) -> int:
    kept = _dry_run_reconstruction()
    if not kept:
        log.error("dry-run: nothing survived the necessity build on the fixture")
        return 2
    if args.limit:
        kept = dict(list(kept.items())[: args.limit])

    backend = StubBackend(_StubReplies(), name="stub")
    outcomes, counts = run_stage_necessity(backend, kept, args.out_dir, None)
    counts.update(necessity_contrasts(outcomes))
    log.info("dry-run necessity: %s",
              json.dumps({k: v for k, v in counts.items() if k != "condition_counts"}))

    summary = {
        "finished_utc": C.stamp_utc(), "dry_run": True, "part": None, "seed": C.SEED,
        "models": [backend.name], "n_items": len(kept),
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "per_model": {backend.name: counts},
    }
    out_path = args.out_dir / "necessity_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


# --------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E3: necessity -- delete the chosen option's own stated attribute and re-ask"
    )
    ap.add_argument("--dry-run", action="store_true",
                     help="run the whole pipeline against a stub model: no GPU, no network")
    ap.add_argument("--self-check", action="store_true",
                     help="item/condition construction and every integrity gate on the "
                          "fixture, no GPU")
    ap.add_argument("--limit", type=int, default=0,
                     help="cap on items processed per model (0 = every gate-clean item)")
    ap.add_argument("--models", nargs="*", default=None,
                     help="substring filter(s) on the committed run's model field")
    ap.add_argument("--part", choices=sorted(PARTS), default="A",
                     help="which run-3 part's committed stage-1 records to read: A=exp3a "
                          "(4 options), B=exp3b (4 options, second roster), C=exp3c (6 options)")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--n-options", type=int, default=None,
                     help="override the part's own experiment_summary.json (rarely needed)")
    ap.add_argument("--scan", type=int, default=None, help="records to scan; default n_items*40")
    ap.add_argument("--data-file", type=Path, default=None,
                     help="local JSON/JSONL dump instead of the hub, for an offline rebuild")
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(args.out_dir / "run_necessity.log")
    log.info("started %s on %s", C.stamp_utc(), platform.platform())

    if args.dry_run:
        return _run_dry(args)

    selected, sel_counts = select_items(
        part=args.part, results_dir=args.results_dir, n_options=args.n_options,
        scan=args.scan, data_file=args.data_file, models=args.models,
    )
    log.info("selected part %s: %s", args.part,
              json.dumps({k: v for k, v in sel_counts.items() if k != "models"}))
    for model, model_counts in sel_counts["models"].items():
        log.info("%s: %s", model, json.dumps(model_counts))

    if args.limit:
        selected = {
            model: dict(list(kept.items())[: args.limit]) for model, kept in selected.items()
        }
    selected = {model: kept for model, kept in selected.items() if kept}
    if not selected:
        log.error("nothing to run: no model in part %s had any item that passed every gate",
                  args.part)
        return 2

    total_items = sum(len(kept) for kept in selected.values())
    check = budget_check(total_items, spent=cumulative_gpu_seconds())
    log.info("budget check: %s", json.dumps(check))
    if not check["ok"]:
        log.error(
            "refusing to start: estimated %.0fs for %d item(s) x %d condition(s) exceeds "
            "the %.0fs remaining of the %.0fs allowance (%.0fs already spent). Lower "
            "--limit, narrow --models, or free budget first.",
            check["estimate_s"], total_items, len(CONDITION_LABELS), check["remaining_s"],
            check["allowance_s"], check["spent_s"],
        )
        return 2

    per_model: dict[str, dict] = {}
    all_outcomes: dict[str, dict[str, bool | None]] = {}
    for model, kept in selected.items():
        try:
            backend = get_backend(model, kind=args.backend, allow_mirror=not args.no_mirror)
        except ModelUnavailable as exc:
            log.error("skipping %s: %s", model, exc)
            per_model[model] = {"error": str(exc)}
            continue

        try:
            outcomes, counts = run_stage_necessity(backend, kept, args.out_dir, args.part)
        finally:
            if backend.kind != "stub":
                backend.close()

        counts.update(necessity_contrasts(outcomes))
        log.info("%s: %s", backend.name,
                  json.dumps({k: v for k, v in counts.items() if k != "condition_counts"}))
        per_model[backend.name] = counts
        for item_id, per_item in outcomes.items():
            all_outcomes[f"{backend.name}::{item_id}"] = per_item

    summary = {
        "finished_utc": C.stamp_utc(), "part": args.part, "dry_run": False, "seed": C.SEED,
        "temperature": C.TEMPERATURE, "models": list(selected), "selection": sel_counts,
        "resolved_n_options": sel_counts["n_options"],
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "pooled": necessity_contrasts(all_outcomes),
        "per_model": per_model,
    }
    out_path = args.out_dir / "necessity_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
