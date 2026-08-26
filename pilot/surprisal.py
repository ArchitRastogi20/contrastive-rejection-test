"""E1: is the sentence inserted into a candidate profile in conditions R1-R4 as unsurprising in
its host paragraph in the content arm (R1/R3, the attribute the model actually named) as it is
in the length-matched arm (R2/R4, a different attribute of the same token length)?

The repair (`pilot.repair`) matches R1/R2 and R3/R4 only on token length before inserting one
sentence into a candidate's profile. Length is not fluency: a sentence that is well-formed and
expected in context can differ sharply in how surprising its individual words are from one that
merely has the same word count. If the two arms are not also matched on that, an effect later
attributed to *content* (does naming the missing fact move the model's choice more than an
unrelated insertion of the same length) could instead be an effect of one arm's sentence simply
reading more fluently -- a confound this experiment exists to rule in or out, not assume away.

**Statistic.** Token-level surprisal of the inserted span only: for each token the model
actually reads there, the negative log-probability it assigned to that token given only the
tokens before it in the rendered profile -- never anything the insertion comes before, since a
causal model cannot have used it anyway. Reported per (item, condition, model) as both the mean
per token and the total, plus the paired R1-R2 and R3-R4 differences.

**The hard part.** `pilot.models` had no prompt-level logprob path before this: `generate()`
produces text, and `letter_probs()` reads the top-k over one *next* token. `Backend.
prompt_token_logprobs` (added in `pilot.models`) is the new one, exposed identically on
`VLLMBackend`, `HFBackend` and `StubBackend` so this module runs against a stub with no GPU.
Locating *which* tokens in the edited prompt are the inserted ones is the correctness-critical
step and is done by `models.find_inserted_span`: the longest common prefix and the longest
common suffix between the unedited and edited prompt's own token sequences, never by searching
for the inserted sentence's text -- a search would be fooled the moment the inserted sentence's
words recur elsewhere in the same profile, which is exactly the case this module's own tests
exercise.

**Not verified without a GPU.** Whether `vllm==0.6.3.post1`'s `SamplingParams` accepts
`prompt_logprobs` at all has not been checked against a real engine anywhere in this project.
`VLLMBackend.prompt_token_logprobs` probes for support the first time it is called and raises
`models.PromptLogprobsUnsupported` -- caught here, not crashed on -- rather than assume the
kwarg works; a real run should fall back to `--backend transformers` if it does not.

**This does not regenerate anything.** The five conditions per item were already built and
already shown to the model in a committed run-3 stage-3 pass (`code/results/exp3a`, `exp3b`,
`exp3c` -- see `pilot.analyze_run3`'s `PARTS`, mirrored here). `repair.build_conditions` is a
pure function of (item, rejection, corpus_index, choice_letter); replaying it against the
already-committed stage-1 response and stage-2 record reproduces byte-identical R0-R4 profiles
without asking a model anything new. This module's own model calls are the *new* instrumentation
(a prompt-logprob read), not a repeat of stage 1-3's generation.

    python -m pilot.surprisal --dry-run                    # whole pipeline, stub model, no GPU
    python -m pilot.surprisal --part A --models Qwen2.5-7B  # the real thing, run-3 Part A
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import config as C
from . import extract, prompts, repair
from .data import Item, build_items, load_records, load_records_from_file
from .extract import Rejection
from .models import (
    Backend,
    ModelUnavailable,
    PromptLogprobs,
    PromptLogprobsUnsupported,
    StubBackend,
    find_inserted_span,
    get_backend,
)
from .run_experiment import bootstrap_ci_mean_diff
from .run_pilot import _StubReplies, _slug
from .watchdog import Watchdog, commit_gpu_seconds, cumulative_gpu_seconds

log = logging.getLogger("surprisal")

# Every condition that inserts a sentence -- R0 is the unedited baseline and has nothing to
# measure surprisal of.
CONDITIONS_WITH_INSERTS = ("R1", "R2", "R3", "R4")
CONTENT_CONTRASTS = (("R1", "R2"), ("R3", "R4"))

# Mirrors `pilot.analyze_run3.PARTS` exactly: run 3's committed stage-1/2/3 output is split
# across these three directories (A, B at 4 options; C at 6). Kept as its own copy here rather
# than imported, since `analyze_run3`'s dict carries a human-readable description string this
# module does not need and importing just for a literal would be a stranger coupling than
# repeating three short strings.
PARTS = {"A": "exp3a", "B": "exp3b", "C": "exp3c"}

# --------------------------------------------------------------------------- GPU budget

# One base (unedited-profile) prompt-logprob read per item, shared across that item's four
# conditions, plus one edited-profile read per condition -- see `run_surprisal`.
CALLS_PER_ITEM = 1 + len(CONDITIONS_WITH_INSERTS)

# Documented, unverified placeholder: there is no GPU available while writing this to time a
# real `prompt_logprobs` call. A single forward pass with no autoregressive decoding over a
# several-hundred-token prompt should cost noticeably less than one of run 3's stage-3 cells
# (a free-text generation of up to `MAX_NEW_TOKENS` plus a one-token letter probe measured at
# roughly 2.6s/cell on Qwen2.5-7B-Instruct, see the experiment ledger) -- 1.0s/call is a
# conservative guess, not a measurement. Replace this constant, and this comment, with what the
# first real run's `gpu_ledger.csv` actually shows.
EST_SECONDS_PER_CALL = 1.0


def estimate_gpu_seconds(n_item_condition_reads: int) -> float:
    """`n_item_condition_reads` x `EST_SECONDS_PER_CALL`.

    `n_item_condition_reads` is `CALLS_PER_ITEM` (one base read plus one per inserted
    condition) times the number of (model, item) pairs about to be processed -- an explicit
    count the caller computes from what it is actually about to run, not a hidden global.
    """
    return n_item_condition_reads * EST_SECONDS_PER_CALL


def budget_check(
    n_item_condition_reads: int, *, spent: float, allowance: float = C.PROJECT_GPU_BUDGET_S,
) -> dict:
    """Whether the estimated cost of `n_item_condition_reads` calls fits in what remains of the
    project's `allowance` GPU-seconds, given `spent` already committed. Pure -- no ledger I/O --
    so it is testable without a real `gpu_ledger.csv`; `main` supplies `spent` from
    `watchdog.cumulative_gpu_seconds()`.
    """
    estimate = estimate_gpu_seconds(n_item_condition_reads)
    remaining = allowance - spent
    return {
        "n_item_condition_reads": n_item_condition_reads, "estimate_s": estimate,
        "spent_s": spent, "allowance_s": allowance, "remaining_s": remaining,
        "ok": estimate <= remaining,
    }


# --------------------------------------------------------------- reconstructing committed items


def _rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _stage1_rows_by_model(root: Path, *, recursive: bool) -> dict[str, list[dict]]:
    """Every committed stage-1 row under `root`, grouped by model.

    `recursive=False` scans only `root` itself (a part's own directory); `recursive=True`
    scans every `stage1_*.jsonl` under `root`. Run 3 Part A (`exp3a`) replayed stage 1 from an
    earlier run's output (`--from-stage1 results/exp2/stage1_...`, see the experiment ledger)
    rather than eliciting it again, so `exp3a` itself carries no `stage1_*.jsonl` of its own --
    the responses this module needs live one directory over. The recursive scan is used only as
    a fallback when a part's own directory has nothing local, so a part that carries its own
    stage 1 (B, C) never pays for it.
    """
    pattern = "**/stage1_*.jsonl" if recursive else "stage1_*.jsonl"
    out: dict[str, list[dict]] = {}
    for path in sorted(root.glob(pattern)):
        for row in _rows(path):
            out.setdefault(row["model"], []).append(row)
    return out


def _stage2_built(exp_dir: Path) -> dict[str, list[dict]]:
    """model -> stage-2 rows that built cleanly (`built: True`) in this directory."""
    out: dict[str, list[dict]] = {}
    for path in sorted(exp_dir.glob("stage2_*.jsonl")):
        for row in _rows(path):
            if row.get("built"):
                out.setdefault(row["model"], []).append(row)
    return out


# `repair.build_conditions_with_diagnostics`'s R3/R4-target selection is gated by
# `config.PREFER_LACKING_TARGET_ABOVE` (`n_options > that threshold`), read as a module
# constant at call time, not passed as an argument -- and it is overridable by the
# `PILOT_PREFER_ABOVE` environment variable (see `config.py`), which run 3 set to 3 rather than
# leaving it at the code's own default of 4 (`config.py`'s own comment: "Set it to 3 to enable
# the preference at 4 options: measured on run 2's own elicitations that recovers roughly a
# third more items (Qwen 95 -> 126)"). Verified directly against this project's own committed
# data, not assumed: exp3a's Qwen stage-2 file records exactly 126 built items, and replaying
# `build_conditions_with_diagnostics` at the code's own default (4) reproduced only 95 --
# rebuilding at 3 reproduces every field of a spot-checked row's diagnostics (`r1_examined`,
# `r2_examined`, `r2_total`, `r3_picked_letter`) exactly. This module cannot import a different
# constant into `repair.py` (a file it does not own), so it overrides the shared `config`
# module's attribute for the duration of one call and restores it immediately after --
# `repair.build_conditions_with_diagnostics` re-reads `config.PREFER_LACKING_TARGET_ABOVE`
# fresh on every call, so this is not a one-time monkeypatch that lingers.
_RUN3_PREFER_LACKING_TARGET_ABOVE = 3


@contextlib.contextmanager
def _prefer_lacking_target_override(value: int):
    original = C.PREFER_LACKING_TARGET_ABOVE
    C.PREFER_LACKING_TARGET_ABOVE = value
    try:
        yield
    finally:
        C.PREFER_LACKING_TARGET_ABOVE = original


def _rejection_from_stage1_row(item: Item, row: dict) -> Rejection | None:
    """Recompute the usable `Rejection` the row's own `response` selected -- one code path,
    the extractor's own, same discipline `run_experiment._selected_rejection` uses."""
    if not row.get("selected"):
        return None
    analysis = extract.analyse(row["response"], item)
    return next((r for r in analysis.rejections if extract.is_usable(item, r)), None)


def _rebuild_item_set(
    *, n_items: int, n_options: int, scan: int, data_file: Path | None,
) -> list[Item]:
    records = (
        load_records_from_file(data_file) if data_file
        else load_records(C.DATASET_ID, C.DATASET_SPLIT, scan)
    )
    return build_items(records, n_items=n_items, n_options=n_options, seed=C.SEED)


# Two different, independently-necessary numbers -- both learned the hard way by cross-checking
# this project's own committed data against `pilot.rescore.rebuild_items` (rule 5) rather than
# by reasoning about the code, and both matter for a faithful reconstruction:
#
# 1. **Corpus breadth** (`corpus_n_items` below): `run_experiment.main` builds one `items` list
#    from *this invocation's own* `--limit` and passes it to `repair.build_corpus_index` a
#    single time, shared across every model that invocation processes -- even under
#    `--from-stage1`, which does not rebuild a bigger corpus_index for the replayed data. A
#    part's own `experiment_summary.json` `n_items` is exactly that `--limit` and is safe to
#    trust for this. Getting this wrong in the *other* direction (too big) is not merely
#    imprecise: R2's candidate pool is drawn from the corpus index and ranked by closest token
#    length, capped at `repair.R2_SEARCH_CAP` candidates; a much larger corpus index than the
#    original run used can push the true historical winner outside that cap and make an item
#    that built cleanly then fail to build now -- measured directly: rebuilding exp3a's Qwen
#    cell against a same-ID-covering-but-oversized 2200-item corpus (see point 2) dropped 31 of
#    126 items with `RepairUnavailable` that the real run's own stage-2 file records as built.
#
# 2. **Item-lookup coverage** (grown below, never just trusted): a part replaying stage 1 via
#    `--from-stage1` iterates `model_items`, rebuilt from *that model's own* replayed stage-1
#    file at `n_items=len(unique ids in it)` (`rescore.rebuild_items`'s own convention) -- a
#    number that can be *larger* than the part's own corpus `--limit` (exp3a's Qwen replayed
#    exp2's 1800-item stage-1 output while exp3a's own corpus was built at 400). Trusting only
#    the part's own `n_items` for lookup as well left 97 of 126 of Qwen's stage-2 items
#    unfindable by id.
#
# `build_items` is deterministic and strictly prefix-growing in `n_items` (a larger `n_items`
# only appends items after the ones a smaller one already found, never changes or reorders
# them), so both needs are met by one growing item list: build enough of it to cover
# `corpus_n_items` (`repair.build_corpus_index` reads exactly the first `corpus_n_items` of it,
# a stable prefix computed once) and separately grow it, per model, until every id that model's
# stage-2 rows reference is present -- never trusting a single recorded number for both jobs.
_N_ITEMS_GROWTH_CAP = 4000


@dataclass
class ReconstructedItem:
    item: Item
    rejection: Rejection
    conditions: dict[str, Item]  # R0-R4, byte-identical to what stage 3 actually rendered


def reconstruct_conditions(
    *,
    part: str,
    results_dir: Path = C.RESULTS_DIR,
    n_options: int | None = None,
    n_items: int | None = None,
    scan: int | None = None,
    data_file: Path | None = None,
    models: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, ReconstructedItem]], dict]:
    """{model: {item_id: ReconstructedItem}} for every stage-2-built item under
    `results_dir / PARTS[part]`, rebuilt deterministically from the committed records -- no
    generation.

    `n_options`/`n_items` default to what that part's own `experiment_summary.json` recorded
    (rule 5: read the real record rather than trust a remembered value); `n_items` there is the
    corpus breadth (see the module-level comment above `_N_ITEMS_GROWTH_CAP` for why this, and
    only this, is safe to trust from the summary while item lookup is not). `--n-options 6` is
    required for Part C only because its own summary already says so.
    """
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; expected one of {sorted(PARTS)}")
    exp_dir = results_dir / PARTS[part]
    summary_path = exp_dir / "experiment_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    )
    resolved_n_options = n_options or summary.get("n_options") or C.N_OPTIONS
    corpus_n_items = n_items or summary.get("n_items") or 400

    stage1_local_by_model = _stage1_rows_by_model(exp_dir, recursive=False)
    stage2_built = _stage2_built(exp_dir)
    stage1_global_by_model: dict[str, list[dict]] | None = None

    counts: dict = {
        "part": part, "exp_dir": str(exp_dir), "n_options": resolved_n_options,
        "corpus_n_items": corpus_n_items, "models": {},
    }

    # One item list, grown as needed and shared across every model in this part -- prefix-stable
    # (see the module-level comment), so growing it for one model never invalidates what an
    # earlier model already looked up. `corpus_index` is computed exactly once, from the first
    # `corpus_n_items` of it, and is never rebuilt from a larger prefix even as `items_list`
    # keeps growing for lookup purposes -- see the module-level comment for why a bigger corpus
    # index is not simply "more thorough" but can silently change which repair gets built.
    items_list: list[Item] = []
    items_by_id: dict[str, Item] = {}
    corpus_index: dict | None = None

    def _grow_to(min_n_items: int) -> None:
        nonlocal items_list, items_by_id, corpus_index
        if len(items_list) >= min_n_items:
            return
        items_list = _rebuild_item_set(
            n_items=min_n_items, n_options=resolved_n_options, scan=min_n_items * 40,
            data_file=data_file,
        )
        items_by_id = {i.item_id: i for i in items_list}
        if corpus_index is None and len(items_list) >= corpus_n_items:
            corpus_index = repair.build_corpus_index(items_list[:corpus_n_items])

    _grow_to(corpus_n_items)  # establishes the corpus baseline before any model is processed

    out: dict[str, dict[str, ReconstructedItem]] = {}
    for model, rows in stage2_built.items():
        if models and not any(m in model for m in models):
            continue
        model_counts = {
            "attempted": 0, "rebuilt": 0, "missing_item": 0, "missing_stage1": 0,
            "option_order_mismatch": 0, "no_usable_rejection": 0, "repair_unavailable": 0,
            "r3_target_verification_failed": 0, "n_items_used": None, "stage1_source": None,
        }

        stage1_rows_for_model = stage1_local_by_model.get(model)
        stage1_source = "local"
        if not stage1_rows_for_model:
            if stage1_global_by_model is None:
                stage1_global_by_model = _stage1_rows_by_model(results_dir, recursive=True)
            stage1_rows_for_model = stage1_global_by_model.get(model)
            stage1_source = "global fallback"
        if not stage1_rows_for_model:
            model_counts["missing_stage1"] = len(rows)
            model_counts["stage1_source"] = "not found"
            out[model] = {}
            counts["models"][model] = model_counts
            continue

        # The same model path can appear in stage-1 files from runs built at a *different*
        # n_options (Part C's exp3c rebuilds every item at 6 options, not 4) -- a row from one
        # of those has an `option_titles` of the wrong length for this part and must never be
        # trusted here, whether it came from the local directory or the global fallback scan.
        # Filtered on the row's own recorded shape (data, not on which directory it came from),
        # since two rows for the same (model, item_id) at the same n_options are expected to
        # agree exactly (`build_item`'s per-item shuffle is seeded by item_id alone, not by how
        # many other items were built alongside it) and are otherwise redundant, not conflicting.
        stage1_rows_for_model = [
            r for r in stage1_rows_for_model
            if len(r.get("option_titles") or []) == resolved_n_options
        ]
        if not stage1_rows_for_model:
            model_counts["missing_stage1"] = len(rows)
            model_counts["stage1_source"] = f"{stage1_source} (found, but at a different n_options)"
            out[model] = {}
            counts["models"][model] = model_counts
            continue
        model_counts["stage1_source"] = stage1_source
        stage1_index_for_model = {r["item_id"]: r for r in stage1_rows_for_model}

        wanted_ids = {row["item_id"] for row in rows}
        _grow_to(max(corpus_n_items, len(stage1_rows_for_model)))
        n = len(items_list)
        while not (wanted_ids <= set(items_by_id)) and n < _N_ITEMS_GROWTH_CAP:
            n = min(_N_ITEMS_GROWTH_CAP, max(n * 2, n + 1))
            _grow_to(n)
        model_counts["n_items_used"] = len(items_list)

        kept: dict[str, ReconstructedItem] = {}
        for row in rows:
            model_counts["attempted"] += 1
            item = items_by_id.get(row["item_id"])
            if item is None:
                model_counts["missing_item"] += 1
                continue

            stage1_row = stage1_index_for_model.get(row["item_id"])
            if stage1_row is None:
                model_counts["missing_stage1"] += 1
                continue

            if stage1_row.get("option_titles") != [o.title for o in item.options]:
                # The rebuilt item's option order does not match what was actually shown --
                # rescoring against it would silently score the wrong profiles. Skip, counted,
                # never trusted anyway.
                model_counts["option_order_mismatch"] += 1
                continue

            rejection = _rejection_from_stage1_row(item, stage1_row)
            if rejection is None:
                model_counts["no_usable_rejection"] += 1
                continue

            try:
                with _prefer_lacking_target_override(_RUN3_PREFER_LACKING_TARGET_ABOVE):
                    conditions, diag = repair.build_conditions_with_diagnostics(
                        item, rejection, corpus_index, stage1_row["choice"],
                        n_options=resolved_n_options,
                    )
            except repair.RepairUnavailable:
                # Deterministic reconstruction disagreeing with a stage-2 "built: true" row
                # would itself be a falsified premise, under this project's stop-and-report
                # rule for falsified premises -- counted rather than silently swallowed, so a
                # run with a nonzero count here is visible in the summary and worth
                # investigating before trusting the rest of the output.
                model_counts["repair_unavailable"] += 1
                continue

            recorded_letter = row.get("r3_picked_letter")
            if recorded_letter is not None and diag.r3_picked_letter != recorded_letter:
                # The R3/R4 target itself does not match what the committed stage-2 row
                # recorded -- using these conditions would silently score against a profile the
                # model never actually saw in stage 3. Dropped and counted, never trusted.
                model_counts["r3_target_verification_failed"] += 1
                continue

            kept[row["item_id"]] = ReconstructedItem(
                item=item, rejection=rejection, conditions=conditions
            )
            model_counts["rebuilt"] += 1

        out[model] = kept
        counts["models"][model] = model_counts

    return out, counts


# --------------------------------------------------------------------------- the statistic


@dataclass
class SpanSurprisal:
    n_tokens: int
    mean_nll: float | None
    total_nll: float | None
    complete: bool
    span_start: int
    span_end: int
    inserted_tokens: list[str]
    per_token_nll: list[float] | None  # raw, kept: mean_nll/total_nll are derived from this


def span_surprisal(base_read: PromptLogprobs, edited_read: PromptLogprobs) -> SpanSurprisal:
    """The mean and total negative log-likelihood of the tokens `find_inserted_span` locates in
    `edited_read` relative to `base_read` -- each token's own logprob, conditioned on nothing
    but the tokens the edited prompt already has before it, which is exactly what
    `edited_read.logprobs` already is (see `PromptLogprobs`).

    `complete` is False whenever the span is empty (no insertion was found -- the two prompts
    were identical, which is itself worth flagging rather than silently reporting 0 tokens as a
    real reading) or any token in it lacks a logprob; such a read must be excluded from the
    paired analysis downstream, never imputed, the same discipline `pilot.models.LetterProbRead`
    already established for the forced-choice read.
    """
    start, end = find_inserted_span(base_read.token_ids, edited_read.token_ids)
    span_logprobs = edited_read.logprobs[start:end]
    inserted_tokens = list(edited_read.tokens[start:end])
    n = end - start

    if n == 0 or any(lp is None for lp in span_logprobs):
        return SpanSurprisal(
            n_tokens=n, mean_nll=None, total_nll=None, complete=False,
            span_start=start, span_end=end, inserted_tokens=inserted_tokens, per_token_nll=None,
        )

    per_token_nll = [-lp for lp in span_logprobs]
    return SpanSurprisal(
        n_tokens=n, mean_nll=sum(per_token_nll) / n, total_nll=sum(per_token_nll),
        complete=True, span_start=start, span_end=end, inserted_tokens=inserted_tokens,
        per_token_nll=per_token_nll,
    )


def condition_summary(rows: list[dict]) -> dict:
    """Per-condition mean of `mean_nll`, over complete reads only."""
    by_condition: dict[str, list[float]] = {c: [] for c in CONDITIONS_WITH_INSERTS}
    for row in rows:
        if row["condition"] in by_condition and row["complete"]:
            by_condition[row["condition"]].append(row["mean_nll"])
    return {
        c: {"n": len(vals), "mean_nll": sum(vals) / len(vals) if vals else None}
        for c, vals in by_condition.items()
    }


def paired_contrast(rows: list[dict], a: str, b: str, *, seed: int) -> dict:
    """Paired mean difference in `mean_nll` between conditions `a` and `b`, one item at a time,
    over items where both reads are complete -- same bootstrap `run_experiment.
    bootstrap_ci_mean_diff` already implements for the choice-probability measure, reused here
    rather than re-derived."""
    by_item: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], {})[row["condition"]] = row

    diffs = []
    for conds in by_item.values():
        ra, rb = conds.get(a), conds.get(b)
        if ra is None or rb is None or not ra["complete"] or not rb["complete"]:
            continue
        diffs.append(ra["mean_nll"] - rb["mean_nll"])

    result = bootstrap_ci_mean_diff(diffs, seed=seed)
    result["contrast"] = f"{a}-{b}"
    return result


# ------------------------------------------------------------------------------- the run itself


def run_surprisal(
    backend: Backend, kept: dict[str, ReconstructedItem], out_dir: Path, part: str,
) -> tuple[list[dict], dict]:
    """One base (R0) prompt-logprob read per item, shared, plus one per inserted condition.
    Writes surprisal_<model>.jsonl and returns the rows plus funnel/condition counts."""
    label = f"{_slug(backend.name)}/surprisal"
    path = out_dir / f"surprisal_{_slug(backend.name)}.jsonl"
    total = len(kept) * CALLS_PER_ITEM
    dog = Watchdog(label=label, total=total, budget_min=C.PER_MODEL_BUDGET_MIN,
                    heartbeat=out_dir / "heartbeat.json")
    dog.sample_vram()

    rows_out: list[dict] = []
    abort_reason: str | None = None

    try:
        with open(path, "a", encoding="utf-8") as fh:
            for item_id, recon in kept.items():
                if dog.should_abort:
                    abort_reason = dog.abort_reason
                    break

                base_chat = prompts.render(recon.conditions["R0"], "elicited")
                base_read = backend.prompt_token_logprobs([base_chat])[0]
                dog.tick()

                for cond in CONDITIONS_WITH_INSERTS:
                    if dog.should_abort:
                        abort_reason = dog.abort_reason
                        break
                    edited_chat = prompts.render(recon.conditions[cond], "elicited")
                    edited_read = backend.prompt_token_logprobs([edited_chat])[0]
                    result = span_surprisal(base_read, edited_read)

                    row = {
                        "model": backend.name, "part": part, "item_id": item_id, "condition": cond,
                        "attribute": recon.rejection.attribute,
                        "rival_letter": recon.rejection.letter,
                        "n_tokens": result.n_tokens, "mean_nll": result.mean_nll,
                        "total_nll": result.total_nll, "complete": result.complete,
                        "span_start": result.span_start, "span_end": result.span_end,
                        "inserted_text": " ".join(result.inserted_tokens),
                        "per_token_nll": result.per_token_nll,  # raw, always kept
                    }
                    rows_out.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    dog.tick()
                if abort_reason:
                    break
    finally:
        pass

    if backend.kind != "stub":
        commit_gpu_seconds(label=label, model=backend.name, backend=backend.kind,
                            items=len(rows_out), seconds=dog.elapsed_s,
                            vram_peak=dog.vram_peak, abort_reason=abort_reason)

    counts = {
        "n_items": len(kept), "n_rows": len(rows_out), "abort_reason": abort_reason,
        "elapsed_min": round(dog.elapsed_s / 60.0, 2), "vram_peak_pct": round(dog.vram_peak * 100, 1),
        "condition_summary": condition_summary(rows_out),
        "contrasts": {
            f"{a}_vs_{b}": paired_contrast(rows_out, a, b, seed=C.SEED)
            for a, b in CONTENT_CONTRASTS
        },
    }
    return rows_out, counts


# --------------------------------------------------------------------------- dry run, no GPU


def _dry_run_reconstruction() -> dict[str, ReconstructedItem]:
    """The same construction `run_experiment.self_check` uses on the fixture, kept to a single
    model's worth of items so `--dry-run` exercises the exact same code path as a real part
    without touching `code/results/` or the network at all."""
    records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
    items = build_items(records, n_items=len(records), n_options=C.N_OPTIONS, seed=C.SEED)
    corpus_index = repair.build_corpus_index(items)
    backend = StubBackend(_StubReplies(), name="stub")

    kept: dict[str, ReconstructedItem] = {}
    for item in items:
        response = backend.generate([prompts.render(item, "elicited")])[0]
        analysis = extract.analyse(response, item)
        rejection = next((r for r in analysis.rejections if extract.is_usable(item, r)), None)
        if rejection is None:
            continue
        try:
            conditions = repair.build_conditions(item, rejection, corpus_index, analysis.choice)
        except repair.RepairUnavailable:
            continue
        kept[item.item_id] = ReconstructedItem(item=item, rejection=rejection, conditions=conditions)
    return kept


def _run_dry(args) -> int:
    kept = _dry_run_reconstruction()
    if not kept:
        log.error("dry-run: nothing survived the repair build on the fixture")
        return 2
    if args.limit:
        kept = dict(list(kept.items())[: args.limit])

    backend = StubBackend(_StubReplies(), name="stub")
    rows, counts = run_surprisal(backend, kept, args.out_dir, None)
    log.info("dry-run surprisal: %s", json.dumps({k: v for k, v in counts.items()
                                                    if k not in ("condition_summary", "contrasts")}))

    summary = {
        "finished_utc": C.stamp_utc(), "dry_run": True, "part": None, "seed": C.SEED,
        "models": [backend.name], "n_items": len(kept),
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "per_model": {backend.name: counts},
    }
    out_path = args.out_dir / "surprisal_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


# --------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E1: inserted-sentence surprisal, content arm vs. length-matched arm"
    )
    ap.add_argument("--dry-run", action="store_true",
                     help="run the whole pipeline against a stub model: no GPU, no network")
    ap.add_argument("--limit", type=int, default=0,
                     help="cap on items processed per model (0 = every stage-2-built item)")
    ap.add_argument("--models", nargs="*", default=None,
                     help="substring filter(s) on the committed run's model field")
    ap.add_argument("--part", choices=sorted(PARTS), default="A",
                     help="which run-3 part's committed records to read: A=exp3a (4 options), "
                          "B=exp3b (4 options, second roster), C=exp3c (6 options)")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR,
                     help="where the committed exp3a/exp3b/exp3c directories live")
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--n-options", type=int, default=None,
                     help="override the part's own experiment_summary.json (rarely needed)")
    ap.add_argument("--n-items", type=int, default=None,
                     help="override the part's own experiment_summary.json (rarely needed)")
    ap.add_argument("--scan", type=int, default=None, help="records to scan; default n_items*40")
    ap.add_argument("--data-file", type=Path, default=None,
                     help="local JSON/JSONL dump instead of the hub, for an offline rebuild")
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(args.out_dir / "surprisal.log")
    log.info("started %s on %s", C.stamp_utc(), platform.platform())

    if args.dry_run:
        return _run_dry(args)

    reconstructed, recon_counts = reconstruct_conditions(
        part=args.part, results_dir=args.results_dir, n_options=args.n_options,
        n_items=args.n_items, scan=args.scan, data_file=args.data_file, models=args.models,
    )
    log.info("reconstructed part %s: %s", args.part,
             json.dumps({k: v for k, v in recon_counts.items() if k != "models"}))
    for model, model_counts in recon_counts["models"].items():
        log.info("%s: %s", model, json.dumps(model_counts))

    if args.limit:
        reconstructed = {
            model: dict(list(kept.items())[: args.limit]) for model, kept in reconstructed.items()
        }
    reconstructed = {model: kept for model, kept in reconstructed.items() if kept}
    if not reconstructed:
        log.error("nothing to run: no model in part %s had any reconstructable item", args.part)
        return 2

    total_reads = sum(len(kept) for kept in reconstructed.values()) * CALLS_PER_ITEM
    check = budget_check(total_reads, spent=cumulative_gpu_seconds())
    log.info("budget check: %s", json.dumps(check))
    if not check["ok"]:
        log.error(
            "refusing to start: estimated %.0fs for %d prompt-logprob call(s) exceeds the "
            "%.0fs remaining of the %.0fs allowance (%.0fs already spent). Lower --limit, "
            "narrow --models, or free budget first.",
            check["estimate_s"], total_reads, check["remaining_s"], check["allowance_s"],
            check["spent_s"],
        )
        return 2

    per_model: dict[str, dict] = {}
    for model, kept in reconstructed.items():
        try:
            backend = get_backend(model, kind=args.backend, allow_mirror=not args.no_mirror)
        except ModelUnavailable as exc:
            log.error("skipping %s: %s", model, exc)
            per_model[model] = {"error": str(exc)}
            continue

        try:
            _rows, counts = run_surprisal(backend, kept, args.out_dir, args.part)
        except PromptLogprobsUnsupported as exc:
            log.error("%s: prompt-level logprobs are not available on this backend: %s",
                       backend.name, exc)
            per_model[backend.name] = {"error": str(exc)}
            continue
        finally:
            if backend.kind != "stub":
                backend.close()

        log.info("%s: %s", backend.name,
                  json.dumps({k: v for k, v in counts.items()
                              if k not in ("condition_summary", "contrasts")}))
        per_model[backend.name] = counts

    summary = {
        "finished_utc": C.stamp_utc(), "part": args.part, "dry_run": False, "seed": C.SEED,
        "models": list(reconstructed), "reconstruction": recon_counts,
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "per_model": per_model,
    }
    out_path = args.out_dir / "surprisal_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
