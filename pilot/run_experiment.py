"""Run the contrastive-repair experiment (S3): does repairing the named defect move the
model's choice more than an unrelated edit does, or more than editing a different profile does?

    python -m pilot.run_experiment --self-check     # stage 1-2 logic + all 8 gates, no GPU
    python -m pilot.run_experiment --dry-run        # whole pipeline, stub model, no GPU
    python -m pilot.run_experiment                  # the real thing

Three stages, each written to its own JSONL with the full raw response kept.

    stage 1  elicit an answer and a contrastive rejection over --limit items, and select the
             items whose rejection is specific, true, and sourceable (`extract.is_usable`).
    stage 2  build R0-R4 for each selected item (`repair.build_conditions`) and run the eight
             integrity gates (`repair.check_integrity`); drops and reasons are counted.
    stage 3  re-ask under all five conditions and record which option was chosen, whether it
             is the repaired rival, and whether the new explanation still names the same defect.

Every field is read off the model's own text by `pilot.extract`; no LLM judges anything here.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import config as C
from . import extract, prompts, repair
from .data import Item, build_items, load_records, load_records_from_file
from .models import (
    Backend,
    ModelUnavailable,
    StubBackend,
    append_letter_probe,
    get_backend,
    preferred_dtype,
    resolve_profile,
)
from .rescore import rebuild_items
from .run_pilot import _StubReplies, _slug, compute_max_len
from .watchdog import Watchdog, commit_gpu_seconds, cumulative_gpu_seconds

log = logging.getLogger("run_experiment")

CONDITION_LABELS = ("R0", "R1", "R2", "R3", "R4")
N_ITEMS_DEFAULT = 400

# The four contrasts the 2x2 design defines (the experiment design doc's amended section): R1-R2
# and R3-R4 are the content effect, at the rival and at the un-complained-about third option;
# R1-R3 and R2-R4 are the location effect, at matched (relevant / irrelevant) content.
CONTINUOUS_CONTRASTS = (("R1", "R2"), ("R3", "R4"), ("R1", "R3"), ("R2", "R4"))
BOOTSTRAP_RESAMPLES = 10_000


# --------------------------------------------------------------------------- stage 1 helpers


def _stage1_row(model: str, backend_kind: str, item: Item, response: str) -> dict:
    """One stage-1 JSONL record: the full raw response, and everything derived from it."""
    analysis = extract.analyse(response, item)
    rejections = []
    usable_letter = None
    usable_attribute = None
    for rej in analysis.rejections:
        usable = extract.is_usable(item, rej)
        rejections.append(
            {
                **asdict(rej),
                "complaint_is_true": extract.complaint_is_true(item, rej),
                "usable": usable,
            }
        )
        if usable and usable_letter is None:
            usable_letter, usable_attribute = rej.letter, rej.attribute

    return {
        "model": model,
        "backend": backend_kind,
        "arm": "elicited",
        "item_id": item.item_id,
        "question": item.question,
        "gold_letter": item.gold_letter,
        "gold_title": item.gold_title,
        "option_titles": [o.title for o in item.options],
        "choice": analysis.choice,
        "choice_correct": None if analysis.choice is None else analysis.choice == item.gold_letter,
        "rejections": rejections,
        "selected": usable_letter is not None,
        "selected_rival_letter": usable_letter,
        "selected_attribute": usable_attribute,
        "response": response,  # always kept
    }


def _selected_rejection(item: Item, row: dict) -> extract.Rejection | None:
    """Recover the Rejection object stage 2 needs from a stage-1 row's raw response.

    Recomputing from `response` (rather than trying to reconstruct from the row's plain-dict
    fields) means a `--from-stage1` re-run always agrees with a fresh run of the same
    extractor version -- there is exactly one code path that builds a Rejection.
    """
    if not row.get("selected"):
        return None
    analysis = extract.analyse(row["response"], item)
    return next((r for r in analysis.rejections if extract.is_usable(item, r)), None)


def run_stage1(backend: Backend, items: list[Item], out_dir: Path) -> tuple[dict[str, dict], dict]:
    """Elicit a response for every item and write stage1_<model>.jsonl. Returns rows by item id
    and the funnel counts."""
    label = f"{_slug(backend.name)}/stage1"
    path = out_dir / f"stage1_{_slug(backend.name)}.jsonl"
    dog = Watchdog(label=label, total=len(items), budget_min=C.PER_MODEL_BUDGET_MIN,
                   heartbeat=out_dir / "heartbeat.json")
    dog.sample_vram()

    rows: dict[str, dict] = {}
    counts = {"items": 0, "unreadable_choice": 0, "usable": 0, "abort_reason": None}
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for item in items:
                if dog.should_abort:
                    counts["abort_reason"] = dog.abort_reason
                    break
                response = backend.generate([prompts.render(item, "elicited")])[0]
                row = _stage1_row(backend.name, backend.kind, item, response)
                rows[item.item_id] = row
                counts["items"] += 1
                if row["choice"] is None:
                    counts["unreadable_choice"] += 1
                if row["selected"]:
                    counts["usable"] += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                dog.tick()
    finally:
        counts["elapsed_min"] = round(dog.elapsed_s / 60.0, 2)
        counts["vram_peak_pct"] = round(dog.vram_peak * 100, 1)
        counts["abort_reason"] = counts["abort_reason"] or dog.abort_reason
        if backend.kind != "stub":
            commit_gpu_seconds(label=label, model=backend.name, backend=backend.kind,
                                items=counts["items"], seconds=dog.elapsed_s,
                                vram_peak=dog.vram_peak, abort_reason=dog.abort_reason)
    return rows, counts


# --------------------------------------------------------------------------- stage 2


_DROP_CATEGORY = (
    ("no sibling profile carries", "no_source_sibling"),
    ("no alternate-attribute source", "no_r2_source"),
    ("no other non-chosen", "no_r3_target"),
    ("R1 sentence is not mechanically", "r1_not_retargetable"),
    ("R2 sentence is not mechanically", "r2_not_retargetable"),
    ("R3 sentence is not mechanically", "r3_not_retargetable"),
    ("R4 sentence is not mechanically", "r4_not_retargetable"),
    ("passed every integrity gate", "integrity_gate_failed"),
    ("is not among this item's options", "rival_not_found"),
    ("is the gold answer", "rival_is_gold"),
    ("choice is unreadable", "unreadable_choice"),
)


def _drop_category(reason: str) -> str:
    for needle, category in _DROP_CATEGORY:
        if needle in reason:
            return category
    return "other_repair_unavailable"


def run_stage2(
    items: list[Item], stage1_rows: dict[str, dict], corpus_index: dict, out_dir: Path, model: str
) -> tuple[dict[str, tuple[Item, extract.Rejection, dict]], dict]:
    """Build and gate-check the five conditions for every stage-1-selected item.

    Returns the items that built cleanly (keyed by item id, with their conditions and stratum)
    plus the funnel/drop counts.
    """
    path = out_dir / f"stage2_{_slug(model)}.jsonl"
    kept: dict[str, tuple[Item, extract.Rejection, dict]] = {}
    drop_reasons: dict[str, int] = {}
    counts = {"attempted": 0, "built": 0, "dropped": 0, "search_cap_hit": 0}

    with open(path, "a", encoding="utf-8") as fh:
        for item in items:
            row = stage1_rows.get(item.item_id)
            if row is None:
                continue
            rejection = _selected_rejection(item, row)
            if rejection is None:
                continue
            counts["attempted"] += 1

            record = {"model": model, "item_id": item.item_id, "rival_title": rejection.title,
                      "attribute": rejection.attribute}
            try:
                # build_conditions_with_diagnostics searches R1 x R2 candidates jointly (see
                # pilot/repair.py) and only raises once both are genuinely exhausted -- the
                # gates have already run internally by the time this returns cleanly, so there
                # is no separate post-hoc check_integrity call here any more.
                conditions, diag = repair.build_conditions_with_diagnostics(
                    item, rejection, corpus_index, row["choice"]
                )
            except repair.RepairUnavailable as exc:
                category = _drop_category(str(exc))
                drop_reasons[category] = drop_reasons.get(category, 0) + 1
                counts["dropped"] += 1
                if exc.cap_hit:
                    counts["search_cap_hit"] += 1
                record.update(built=False, drop_reason=str(exc), drop_category=category,
                              gate_failures=exc.gate_failures, cap_hit=exc.cap_hit)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            counts["built"] += 1
            if diag.cap_hit:
                counts["search_cap_hit"] += 1
            record.update(built=True, drop_reason=None, drop_category=None, stratum=diag.stratum,
                          r1_examined=diag.r1_examined, r2_examined=diag.r2_examined,
                          r1_total=diag.r1_total, r2_total=diag.r2_total, cap_hit=diag.cap_hit,
                          r3_candidates_total=diag.r3_candidates_total,
                          r3_candidates_without_attribute=diag.r3_candidates_without_attribute,
                          r3_picked_letter=diag.r3_picked_letter)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept[item.item_id] = (item, rejection, conditions)

    counts["drop_reasons"] = drop_reasons
    return kept, counts


# --------------------------------------------------------------------------- stage 3


def run_stage3(
    backend: Backend, kept: dict[str, tuple[Item, extract.Rejection, dict]], out_dir: Path
) -> tuple[dict, dict]:
    """Re-ask under R0-R4 for every gate-clean item. Returns per-item per-condition outcomes
    and the funnel/condition counts.

    Two backend calls per condition: the existing free-text answer-and-explanation call
    (unchanged -- `choice` / `chosen_is_edited` etc. are exactly as they were), plus a second,
    additional forced-choice probability read on the same rendered context (see
    `models.append_letter_probe`). The second call's per-item, per-condition results are
    returned inside `counts["prob_outcomes"]` rather than folded into `outcomes`, so
    `mcnemar()` and every existing caller of `outcomes` sees exactly what it saw before.
    """
    label = f"{_slug(backend.name)}/stage3"
    path = out_dir / f"stage3_{_slug(backend.name)}.jsonl"
    total = len(kept) * len(CONDITION_LABELS) * 2  # free-text call + letter-probe call, each
    dog = Watchdog(label=label, total=total, budget_min=C.PER_MODEL_BUDGET_MIN,
                    heartbeat=out_dir / "heartbeat.json")
    dog.sample_vram()

    outcomes: dict[str, dict[str, bool | None]] = {}
    prob_outcomes: dict[str, dict[str, dict]] = {}
    condition_counts = {c: {"chosen_rival": 0, "unreadable": 0, "n": 0} for c in CONDITION_LABELS}
    abort_reason = None

    try:
        with open(path, "a", encoding="utf-8") as fh:
            for item_id, (item, rejection, conditions) in kept.items():
                rival_letter = rejection.letter
                # The letter R3/R4 edit -- read off the constructed items by the same
                # append-only diff `check_integrity` uses, rather than re-deriving the
                # exclusion rule a second time. R0 edits nothing, so its edited letter is None,
                # not a guess at what R3/R4 touch (see r3_recency.py for why that distinction
                # matters for the "before the edit" baseline reading).
                r3_idx, _ = repair._edited_slot(item, conditions["R3"])
                r3_letter = chr(ord("A") + r3_idx) if r3_idx is not None else None
                edited_letter_by_condition = {
                    "R0": None, "R1": rival_letter, "R2": rival_letter,
                    "R3": r3_letter, "R4": r3_letter,
                }

                per_item: dict[str, bool | None] = {}
                per_item_probs: dict[str, dict] = {}
                # R0's own read, captured the moment it is processed (R0 is always first in
                # CONDITION_LABELS) and reused as the baseline for every other condition's
                # delta on *its own* target letter -- see `delta_from_baseline`.
                r0_probs: dict[str, float] = {}
                r0_complete = False
                for label_c in CONDITION_LABELS:
                    if dog.should_abort:
                        abort_reason = dog.abort_reason
                        break
                    cond_item = conditions[label_c]
                    chat = prompts.render(cond_item, "elicited")
                    response = backend.generate([chat])[0]
                    analysis = extract.analyse(response, cond_item)
                    chosen_rival = (
                        None if analysis.choice is None else analysis.choice == rival_letter
                    )
                    same_defect = any(
                        r.letter == rival_letter and r.attribute == rejection.attribute
                        for r in analysis.rejections
                    )
                    edited_letter = edited_letter_by_condition[label_c]
                    chosen_is_edited = (
                        None if analysis.choice is None or edited_letter is None
                        else analysis.choice == edited_letter
                    )
                    per_item[label_c] = chosen_rival

                    condition_counts[label_c]["n"] += 1
                    if analysis.choice is None:
                        condition_counts[label_c]["unreadable"] += 1
                    elif chosen_rival:
                        condition_counts[label_c]["chosen_rival"] += 1

                    # The second, additional call: same rendered context, one appended
                    # instruction, forced to a single token. Never touches anything above.
                    candidates = [chr(ord("A") + i) for i in range(len(cond_item.options))]
                    probe_chat = append_letter_probe(chat)
                    letter_read = backend.letter_probs([probe_chat], [candidates])[0]
                    p_edited = (
                        None if edited_letter is None else letter_read.probs.get(edited_letter)
                    )
                    if label_c == "R0":
                        r0_probs, r0_complete = letter_read.probs, letter_read.complete

                    # The baseline-corrected quantity the four continuous contrasts must use
                    # instead of raw `p_edited` for R1-R3/R2-R4 (see `delta_from_baseline`):
                    # P(edited letter | this condition) - P(*that same* edited letter | R0).
                    # None whenever either read is incomplete -- excluded downstream, not
                    # imputed -- and always None for R0 itself, which has no edited letter.
                    r0_p_target = None
                    delta_p_edited = None
                    if edited_letter is not None and letter_read.complete and r0_complete:
                        r0_p_target = r0_probs.get(edited_letter)
                        if r0_p_target is not None and p_edited is not None:
                            delta_p_edited = p_edited - r0_p_target

                    per_item_probs[label_c] = {
                        "p_edited": p_edited,
                        "complete": letter_read.complete,
                        "r0_complete": r0_complete,
                        "r0_p_target": r0_p_target,
                        "delta_p_edited": delta_p_edited,
                    }

                    fh.write(json.dumps({
                        "model": backend.name, "item_id": item_id, "condition": label_c,
                        "rival_letter": rival_letter, "attribute": rejection.attribute,
                        "choice": analysis.choice, "chosen_is_repaired_rival": chosen_rival,
                        "still_names_same_defect": same_defect,
                        # New fields, added alongside the ones above rather than replacing them
                        # -- run 1's records never have these and must stay comparable.
                        "edited_letter": edited_letter, "chosen_is_edited": chosen_is_edited,
                        # The forced-choice probability read: additive, second measure.
                        "letter_probe": letter_read.to_dict(), "p_edited": p_edited,
                        # The baseline-corrected version -- see `delta_from_baseline` above.
                        "r0_p_target": r0_p_target, "delta_p_edited": delta_p_edited,
                        "response": response,  # always kept
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    dog.tick(2)  # the free-text call and the letter-probe call
                outcomes[item_id] = per_item
                prob_outcomes[item_id] = per_item_probs
                if abort_reason:
                    break
    finally:
        pass

    if backend.kind != "stub":
        commit_gpu_seconds(label=label, model=backend.name, backend=backend.kind,
                            items=len(outcomes), seconds=dog.elapsed_s,
                            vram_peak=dog.vram_peak, abort_reason=abort_reason)

    counts = {"n_items": len(outcomes), "condition_counts": condition_counts,
              "abort_reason": abort_reason, "elapsed_min": round(dog.elapsed_s / 60.0, 2),
              # Per-item, per-condition P(edited option) and completeness -- kept out of
              # `outcomes` itself so `mcnemar()` and every existing reader of `outcomes` is
              # unaffected. The caller pops this before logging/serialising the rest of
              # `counts`, since it is per-item detail already fully present in the JSONL file.
              "prob_outcomes": prob_outcomes}
    return outcomes, counts


# --------------------------------------------------------------------------- McNemar, no scipy

# scipy is not guaranteed on this image (code/requirements.txt does not pin it), so the exact
# two-sided McNemar p-value is computed from `math.comb` alone: it is a two-sided exact
# binomial test on the discordant pairs at p=0.5, which is what McNemar's exact test is.


def mcnemar(outcomes: dict[str, dict[str, bool | None]], a: str, b: str) -> dict:
    """Discordant-pair counts b, c and the exact two-sided binomial p for conditions a vs b."""
    disc_a_only = 0  # a True, b False
    disc_b_only = 0  # a False, b True
    for per_item in outcomes.values():
        va, vb = per_item.get(a), per_item.get(b)
        if va is None or vb is None or va == vb:
            continue
        if va and not vb:
            disc_a_only += 1
        elif vb and not va:
            disc_b_only += 1

    n = disc_a_only + disc_b_only
    if n == 0:
        p = 1.0
    else:
        k = min(disc_a_only, disc_b_only)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)

    return {"b_a_only": disc_a_only, "c_b_only": disc_b_only, "n_discordant": n,
            "p_exact_two_sided": p}


# --------------------------------------------------------- the continuous measure's bootstrap
#
# No scipy or numpy: a seeded `random.Random` resample, same discipline as `mcnemar` above
# using `math.comb` instead of scipy. The seed is recorded in every result so it is exactly
# reproducible offline from the per-item `p_edited`/`complete` fields the JSONL already has.


def bootstrap_ci_mean_diff(
    diffs: Sequence[float], *, seed: int, n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
) -> dict:
    """95% CI for the mean of `diffs` by the percentile bootstrap. Deterministic: the same
    `diffs` and `seed` always resample identically, because `random.Random(seed)` is drawn from
    in a fixed order, one full resample of length `n` at a time."""
    n = len(diffs)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0,
                "seed": seed, "n_resamples": n_resamples}

    mean = sum(diffs) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += diffs[rng.randrange(n)]
        means.append(resample_sum / n)
    means.sort()

    lo_idx = min(n_resamples - 1, max(0, int((alpha / 2) * n_resamples)))
    hi_idx = min(n_resamples - 1, max(0, int((1 - alpha / 2) * n_resamples) - 1))
    return {"mean": mean, "ci_low": means[lo_idx], "ci_high": means[hi_idx], "n": n,
            "seed": seed, "n_resamples": n_resamples}


def delta_from_baseline(
    condition_probs: dict[str, float], baseline_probs: dict[str, float], letter: str | None
) -> float | None:
    """P(`letter` given this condition) minus P(`letter` given R0) -- the same letter on both
    sides.

    This, not raw `p_edited`, is what the R1-R3 and R2-R4 *location* contrasts must
    difference. R1/R2 edit the rival the model just disparaged; R3/R4 edit a third option it
    said nothing about. Those two options do not start from the same baseline probability, so
    differencing their raw `p_edited` values would measure that baseline gap as much as any
    effect of the edit -- exactly the confound that forced the McNemar analysis to hedge its
    location claim (see the experiment design doc's correction section).
    Subtracting each condition's own R0 baseline, on its own target letter, before comparing
    across location removes that gap: R1's delta and R3's delta are both changes from their
    own item's unedited read, so comparing the deltas is a genuine location comparison. R1-R2
    and R3-R4 (the content contrasts) do not have this problem -- both sides already share one
    target -- but the same delta is reported for them too, for uniformity and because it is
    free.

    Caveat this bounds rather than solves: a probability already near 0 or 1 has less room
    left to move than one nearer the middle, so an item whose rival started at a near-floor R0
    probability and one whose third option started near the middle are still not perfectly
    comparable just because both are now expressed as deltas. See `r0_baseline_diagnostic`
    below for a way to inspect whether that is actually happening on a given run's data,
    rather than assuming it away.
    """
    if letter is None:
        return None
    a = condition_probs.get(letter)
    b = baseline_probs.get(letter)
    if a is None or b is None:
        return None
    return a - b


def continuous_contrast(
    prob_outcomes: dict[str, dict[str, dict]], a: str, b: str, *, seed: int,
    measure: str = "delta_p_edited",
) -> dict:
    """Paired mean difference in `measure` between conditions `a` and `b`, over every item
    where that value is present on both sides -- an incomplete read (this condition's own, or
    the R0 baseline a delta needs) is excluded from this, never imputed, per the experiment
    design doc's addendum, and separately counted by reason below.

    `measure` defaults to "delta_p_edited" (the baseline-corrected quantity `run_stage3`
    already computed per item -- see `delta_from_baseline`), which is what the four official
    contrasts must use. "p_edited", the raw quantity, is also supported so the two can be
    reported side by side and the difference stays inspectable -- see
    `all_continuous_contrasts`. 95% CI by seeded bootstrap.
    """
    diffs = []
    excluded_incomplete_condition = 0
    excluded_incomplete_r0 = 0
    excluded_other = 0
    for per_item in prob_outcomes.values():
        ra, rb = per_item.get(a), per_item.get(b)
        if ra is None or rb is None:
            continue
        va, vb = ra.get(measure), rb.get(measure)
        if va is None or vb is None:
            if not ra.get("complete", True) or not rb.get("complete", True):
                excluded_incomplete_condition += 1
            elif not ra.get("r0_complete", True) or not rb.get("r0_complete", True):
                excluded_incomplete_r0 += 1
            else:
                excluded_other += 1
            continue
        diffs.append(va - vb)

    result = bootstrap_ci_mean_diff(diffs, seed=seed)
    result.update(
        contrast=f"{a}-{b}", measure=measure, n_pairs_total=len(prob_outcomes),
        n_pairs_complete=len(diffs),
        n_excluded_incomplete_condition=excluded_incomplete_condition,
        n_excluded_incomplete_r0=excluded_incomplete_r0,
        n_excluded_other=excluded_other,
    )
    return result


def all_continuous_contrasts(prob_outcomes: dict[str, dict[str, dict]], *, seed: int) -> dict:
    """Both measures, all four contrasts. "delta_p_edited" is the one the design calls for
    (see `delta_from_baseline`); "p_edited_raw" is kept alongside it purely so a reader can see
    how much, if at all, the baseline correction changes the answer -- it should not for
    R1-R2 / R3-R4, where both sides already share a target, and it can for R1-R3 / R2-R4,
    where they do not."""
    return {
        "delta_p_edited": {
            f"{a}_vs_{b}": continuous_contrast(prob_outcomes, a, b, seed=seed,
                                                measure="delta_p_edited")
            for a, b in CONTINUOUS_CONTRASTS
        },
        "p_edited_raw": {
            f"{a}_vs_{b}": continuous_contrast(prob_outcomes, a, b, seed=seed,
                                                measure="p_edited")
            for a, b in CONTINUOUS_CONTRASTS
        },
    }


def r0_baseline_diagnostic(prob_outcomes: dict[str, dict[str, dict]]) -> dict:
    """The R0 (unedited) baseline probability of each location's own target letter --
    R1/R2's target (the rival) versus R3/R4's target (the un-complained-about third option) --
    diagnostic only, not a hypothesis test. Read only off R1 and R3 (R2/R4 share the same
    target and the same R0 read as R1/R3 respectively, so including them would just duplicate
    every item's number rather than add information). If these two ranges are far apart, a
    reader should not take the delta-based location contrast above as having fully equalised
    the two locations -- see `delta_from_baseline`'s caveat."""
    groups: dict[str, list[float]] = {
        "rival_target_R1_R2": [], "third_option_target_R3_R4": [],
    }
    for per_item in prob_outcomes.values():
        r1 = per_item.get("R1")
        if r1 is not None and r1.get("r0_p_target") is not None:
            groups["rival_target_R1_R2"].append(r1["r0_p_target"])
        r3 = per_item.get("R3")
        if r3 is not None and r3.get("r0_p_target") is not None:
            groups["third_option_target_R3_R4"].append(r3["r0_p_target"])

    out = {}
    for group, values in groups.items():
        if values:
            out[group] = {"n": len(values), "mean": sum(values) / len(values),
                           "min": min(values), "max": max(values)}
        else:
            out[group] = {"n": 0, "mean": None, "min": None, "max": None}
    return out


# --------------------------------------------------------------------------- self-check


def _fixture_item_for_gates() -> tuple[Item, extract.Rejection]:
    from .data import Entity

    item = Item(
        item_id="self-check-1",
        question="Who directed the film Blue River?",
        answer="Anna Kowalska",
        gold_title="Anna Kowalska",
        options=[
            Entity("Anna Kowalska", ["Anna Kowalska is a Polish filmmaker.",
                                      "She was born in 1970 in Krakow.",
                                      "She directed Blue River."]),
            Entity("Bruno Kowalski", ["Bruno Kowalski is a Polish cinematographer.",
                                       "He worked on several feature films."]),
            Entity("Clara Novak", ["Clara Novak is a Czech screenwriter.",
                                    "Her mother was named Maria Elena Novak."]),
            Entity("Dawid Kowalski", ["Dawid Kowalski is a Polish film editor.",
                                       "He edited three documentaries."]),
        ],
        relations=[("Blue River", "director", "Anna Kowalska"),
                   ("Anna Kowalska", "date of birth", "1970")],
    )
    rejection = extract.Rejection(
        sentence="It is not Bruno Kowalski, because his profile never gives a date of birth.",
        letter="B", title="Bruno Kowalski", matched_by="title", attribute="date_of_birth",
        cue="birth", negation_cue="not",
    )
    return item, rejection


def _gate_violation_problems() -> list[str]:
    """Deliberately break each of the eight gates and confirm `check_integrity` catches it.

    Independent of `test_repair.py`: this runs from the command line with no pytest and no
    GPU, so a regression in the gates cannot pass silently just because a particular run's
    data never happened to exercise one of them.
    """
    import copy

    item, rejection = _fixture_item_for_gates()
    corpus_index = repair.build_corpus_index([item])
    conditions = repair.build_conditions(item, rejection, corpus_index, choice_letter="A")
    baseline_failures = repair.check_integrity(conditions, item, rejection)
    problems = []
    if baseline_failures:
        problems.append(f"a clean build already fails: {baseline_failures}")
        return problems

    def gate_fires(label: str, broken: dict, prefix: str) -> str | None:
        failures = repair.check_integrity(broken, item, rejection)
        if not any(f.startswith(prefix) for f in failures):
            return f"{label}: expected a failure starting with {prefix!r}, got {failures}"
        return None

    # gate 1: undo the R1 edit so the attribute is still absent
    broken = copy.deepcopy(conditions)
    broken["R1"].options[1].sentences.pop()
    if (p := gate_fires("gate1", broken, "gate1:")) is not None:
        problems.append(p)

    # gate 2: make R2 also repair the named attribute
    broken = copy.deepcopy(conditions)
    broken["R2"].options[1].sentences.append("Bruno Kowalski was born in 1970 in Krakow.")
    if (p := gate_fires("gate2", broken, "gate2:")) is not None:
        problems.append(p)

    # gate 3: make R2's inserted sentence far longer than R1's
    broken = copy.deepcopy(conditions)
    broken["R2"].options[1].sentences[-1] = (
        broken["R2"].options[1].sentences[-1] + " " + "word " * 40
    )
    if (p := gate_fires("gate3", broken, "gate3:")) is not None:
        problems.append(p)

    # gate 4: insert a date of birth after an existing date of death
    broken = copy.deepcopy(conditions)
    broken["R1"].options[1].sentences[-1] = "Bruno Kowalski was born in 1999."
    broken["R1"].options[1].sentences.append("Bruno Kowalski died in 1950.")
    if (p := gate_fires("gate4", broken, "gate4:")) is not None:
        problems.append(p)

    # gate 5: the inserted sentence names another candidate
    broken = copy.deepcopy(conditions)
    broken["R1"].options[1].sentences[-1] = "Bruno Kowalski was born the same year as Clara Novak."
    if (p := gate_fires("gate5", broken, "gate5:")) is not None:
        problems.append(p)

    # gate 6: touch the gold profile
    broken = copy.deepcopy(conditions)
    broken["R1"].options[0].sentences.append("This sentence should never be here.")
    if (p := gate_fires("gate6", broken, "gate6:")) is not None:
        problems.append(p)

    # gate 7: reorder the options in one condition
    broken = copy.deepcopy(conditions)
    broken["R1"].options[0], broken["R1"].options[1] = (
        broken["R1"].options[1], broken["R1"].options[0]
    )
    if (p := gate_fires("gate7", broken, "gate7:")) is not None:
        problems.append(p)

    # gate 8: make R4 also repair the named attribute, at the option it shares with R3
    # (Clara Novak, index 2) rather than the rival's -- R4's counterpart of gate 2.
    broken = copy.deepcopy(conditions)
    broken["R4"].options[2].sentences.append("Clara Novak was born in 1970 in Krakow.")
    if (p := gate_fires("gate8", broken, "gate8:")) is not None:
        problems.append(p)

    return problems


def self_check() -> int:
    """Stage 1-2 logic on the fixture, plus all eight integrity gates -- no GPU, no network.
    Exits non-zero on any failure, the equivalent of the previous study's `--self-check`."""
    C.setup_logging()
    records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
    items = build_items(records, n_items=len(records), n_options=C.N_OPTIONS, seed=C.SEED)
    if not items:
        log.error("self-check: no items built from the fixture")
        return 2

    corpus_index = repair.build_corpus_index(items)
    backend = StubBackend(_StubReplies(), name="self-check-stub")

    problems: list[str] = []
    built, dropped = 0, 0
    for item in items:
        response = backend.generate([prompts.render(item, "elicited")])[0]
        analysis = extract.analyse(response, item)
        usable = next((r for r in analysis.rejections if extract.is_usable(item, r)), None)
        if usable is None:
            continue
        try:
            conditions = repair.build_conditions(item, usable, corpus_index, analysis.choice)
        except repair.RepairUnavailable:
            dropped += 1
            continue
        failures = repair.check_integrity(conditions, item, usable)
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


# --------------------------------------------------------------------------- main


def _source_name(args) -> str:
    if args.data_file:
        return str(args.data_file)
    if args.from_stage1:
        return f"stage1 replay: {args.from_stage1}"
    if args.dry_run:
        return "fixture:twowiki_sample.jsonl"
    return C.DATASET_ID


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Contrastive-repair experiment (S3)")
    ap.add_argument("--dry-run", action="store_true",
                     help="run the whole pipeline against a stub model: no GPU, no network")
    ap.add_argument("--self-check", action="store_true",
                     help="stage 1-2 logic and every integrity gate on the fixture, no GPU")
    ap.add_argument("--limit", type=int, default=N_ITEMS_DEFAULT)
    ap.add_argument("--scan", type=int, default=0,
                     help="records to scan for usable items (default: 40x limit)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--profile", default="auto",
                     help="model roster: auto | 3090ti | t4. auto picks by the card's VRAM")
    ap.add_argument("--n-options", type=int, default=C.N_OPTIONS,
                    help="candidate options per item. 4 reproduces runs 1-2 exactly; "
                         "above 4 enables the preference for an R3/R4 target that "
                         "already lacks the named attribute, which relieves gate 8")
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--data-file", type=Path, default=None,
                     help="local JSON/JSONL dump instead of the hub")
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--stage1-only", action="store_true",
                     help="elicit and select, then stop -- no repair, no generation")
    ap.add_argument("--from-stage1", type=Path, default=None,
                     help="a stage1_*.jsonl from a previous run: skip re-eliciting, go straight "
                          "to stage 2/3 using its responses")
    ap.add_argument("--stage2-only", action="store_true",
                     help="with --from-stage1: build+gate stage 2 alone and stop -- no stage 3, "
                          "no model backend is ever loaded, no GPU. Rebuilds the item profiles "
                          "needed for the repair from the dataset (a data fetch, not a model "
                          "call); pass --data-file for a fully offline rebuild from a local "
                          "dump. Requires --out-dir to point outside results/exp/, which is a "
                          "committed scientific record and must not be modified in place.")
    args = ap.parse_args(argv)

    if args.stage2_only and not args.from_stage1:
        ap.error("--stage2-only requires --from-stage1 (there is no stage 1 to build on)")
    protected_dir = (C.RESULTS_DIR / "exp").resolve()
    if args.out_dir.resolve() == protected_dir:
        ap.error(f"--out-dir must not be {protected_dir}: results/exp/ is a committed "
                  "scientific record and this run must write somewhere else, e.g. "
                  "results/exp_rebuild")

    if args.self_check:
        return self_check()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(args.out_dir / "run_experiment.log")
    log.info("started %s on %s", C.stamp_utc(), platform.platform())

    scan = args.scan or args.limit * 40
    if args.data_file:
        records = load_records_from_file(args.data_file)
    elif args.dry_run:
        records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
    else:
        log.info("loading %s [%s], scanning %d records", C.DATASET_ID, C.DATASET_SPLIT, scan)
        records = load_records(C.DATASET_ID, C.DATASET_SPLIT, scan)

    items = build_items(records, n_items=args.limit, n_options=args.n_options, seed=C.SEED)
    log.info("built %d items from %d records", len(items), len(records))
    if not items:
        log.error("no usable items. Run pilot.run_pilot --inspect-schema first.")
        return 2

    corpus_index = repair.build_corpus_index(items)
    max_len = compute_max_len(items)
    profile = resolve_profile(args.profile)
    if profile not in C.MODEL_PROFILES:
        log.error("unknown profile %r; expected one of %s", profile, sorted(C.MODEL_PROFILES))
        return 2

    # Same source `rebuild_items` (stage 2/3) should reload records from as the main path just
    # used above -- otherwise --from-stage1 combined with --dry-run silently falls through to
    # a live hub call instead of the fixture, which is exactly the kind of thing --dry-run
    # promises never to do.
    rebuild_data_file = args.data_file or (
        C.FIXTURE_DIR / "twowiki_sample.jsonl" if args.dry_run else None
    )

    from_stage1_rows: dict[str, list[dict]] = {}
    if args.from_stage1:
        raw = [json.loads(l) for l in args.from_stage1.read_text(encoding="utf-8").splitlines()
               if l.strip()]
        if not raw:
            log.error("no rows in %s", args.from_stage1)
            return 2
        for row in raw:
            from_stage1_rows.setdefault(row["model"], []).append(row)
        model_names = sorted(from_stage1_rows)
        log.info("replaying stage 1 from %s for model(s) %s", args.from_stage1, model_names)
    else:
        model_names = args.models or (["stub"] if args.dry_run else C.MODEL_PROFILES[profile])

    summary: dict[str, dict] = {}
    all_drop_reasons: dict[str, int] = {}
    # Keyed "{model}::{item_id}" so items (built once, shared across models) don't collide
    # when pooled: each model's read of the same item is its own paired observation.
    all_prob_outcomes: dict[str, dict[str, dict]] = {}

    for name in model_names:
        if args.from_stage1:
            rows = from_stage1_rows[name]
            wanted_ids = []
            for r in rows:
                if r["item_id"] not in wanted_ids:
                    wanted_ids.append(r["item_id"])
            rebuilt = rebuild_items(rows, scan, rebuild_data_file)
            model_items = [rebuilt[i] for i in wanted_ids]
            stage1_rows = {r["item_id"]: r for r in rows}
            stage1_counts = {
                "items": len(rows), "usable": sum(1 for r in rows if r["selected"]),
                "unreadable_choice": sum(1 for r in rows if r["choice"] is None),
                "replayed_from": str(args.from_stage1),
            }
            backend_name = name
            backend = None
        else:
            model_items = items
            if args.dry_run:
                backend = StubBackend(_StubReplies(), name="stub")
            else:
                try:
                    backend = get_backend(name, kind=args.backend, max_model_len=max_len,
                                           allow_mirror=not args.no_mirror)
                except ModelUnavailable as exc:
                    log.error("skipping %s: %s", name, exc)
                    summary[name] = {"error": str(exc)}
                    continue
            if backend.name != name:
                log.warning("running %s in place of %s", backend.name, name)
            log.info("%s ready via the %s backend", backend.name, backend.kind)
            stage1_rows, stage1_counts = run_stage1(backend, model_items, args.out_dir)
            backend_name = backend.name

        log.info("%s stage1: %s", backend_name, json.dumps(stage1_counts))
        model_summary = {"stage1": stage1_counts}

        if not args.stage1_only:
            kept, stage2_counts = run_stage2(model_items, stage1_rows, corpus_index,
                                              args.out_dir, backend_name)
            log.info("%s stage2: %s", backend_name,
                      json.dumps({k: v for k, v in stage2_counts.items() if k != "drop_reasons"}))
            model_summary["stage2"] = stage2_counts
            for reason, n in stage2_counts["drop_reasons"].items():
                all_drop_reasons[reason] = all_drop_reasons.get(reason, 0) + n

            if args.stage2_only:
                log.info("%s: --stage2-only, stopping after stage 2 (%d item(s) kept, no "
                         "backend loaded)", backend_name, len(kept))
            elif kept:
                if backend is None:  # --from-stage1: still need a live backend for stage 3
                    if args.dry_run:
                        backend = StubBackend(_StubReplies(), name="stub")
                    else:
                        try:
                            backend = get_backend(name, kind=args.backend, max_model_len=max_len,
                                                   allow_mirror=not args.no_mirror)
                        except ModelUnavailable as exc:
                            log.error("stage 3 needs %s but it could not be loaded: %s", name, exc)
                            summary[name] = model_summary
                            continue
                outcomes, stage3_counts = run_stage3(backend, kept, args.out_dir)
                # Per-item continuous detail, popped out before logging/serialising the rest
                # of stage3_counts -- it is per-item, already fully present in the JSONL file.
                prob_outcomes = stage3_counts.pop("prob_outcomes", {})
                log.info("%s stage3: %s", backend.name, json.dumps(stage3_counts))
                model_summary["stage3"] = stage3_counts
                model_summary["mcnemar_r1_vs_r2"] = mcnemar(outcomes, "R1", "R2")
                model_summary["mcnemar_r1_vs_r3"] = mcnemar(outcomes, "R1", "R3")
                # R3/R4's own 2x2 contrasts are read from `chosen_is_edited`, not from
                # `chosen_is_repaired_rival` (which `mcnemar`/`outcomes` carry and which is not
                # what R3/R4 land on) -- that reading is r3_recency.py's job, offline, same as
                # it already is for run 1. Not duplicated here.

                # The continuous measure: paired mean difference in `delta_p_edited` (primary)
                # and raw `p_edited` (for inspection), same four contrasts, bootstrap 95% CI,
                # alongside (not instead of) McNemar above.
                model_summary["continuous"] = all_continuous_contrasts(prob_outcomes, seed=C.SEED)
                # Diagnostic, not a test: are R1/R2's target and R3/R4's target starting from
                # comparable R0 baselines? See `delta_from_baseline`'s caveat.
                model_summary["r0_baseline_by_target"] = r0_baseline_diagnostic(prob_outcomes)
                for item_id, per_item_probs in prob_outcomes.items():
                    all_prob_outcomes[f"{backend.name}::{item_id}"] = per_item_probs
            else:
                log.warning("%s: nothing survived stage 2, skipping stage 3", backend_name)

        summary[backend_name] = model_summary
        if backend is not None and backend.kind != "stub":
            backend.close()

    payload = {
        "finished_utc": C.stamp_utc(),
        "dataset": _source_name(args),
        "split": C.DATASET_SPLIT,
        "seed": C.SEED,
        "temperature": C.TEMPERATURE,
        "max_model_len": max_len,
        "dtype": preferred_dtype(),
        "profile": profile,
        "n_items": len(items),
        "n_options": args.n_options,
        "dry_run": args.dry_run,
        "stage1_only": args.stage1_only,
        "stage2_only": args.stage2_only,
        "drop_reasons": all_drop_reasons,
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "continuous_pooled": all_continuous_contrasts(all_prob_outcomes, seed=C.SEED),
        "r0_baseline_by_target_pooled": r0_baseline_diagnostic(all_prob_outcomes),
        "per_model": summary,
    }
    (args.out_dir / "experiment_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("wrote %s", args.out_dir / "experiment_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
