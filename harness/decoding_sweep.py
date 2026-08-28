"""E4: does the effect survive sampling, or only greedy decoding?

Every result this design has produced so far comes from temperature-0 decoding: exactly
reproducible, but silent on whether a repaired rival's pull on the model's choice is a stable
property of the distribution the model is sampling from, or an artefact of whichever single
continuation greedy decoding happens to walk down. This module re-runs stage 3 -- re-asking
under R0-R4 -- at one or more temperatures with N samples per item per condition, and reports:
how the choice distribution looks at each temperature, what the modal (most frequent) choice per
item is, how often that modal choice agrees with the already-committed greedy result, and
whether the four paired contrasts (`run_experiment.CONTINUOUS_CONTRASTS`) still hold when
recomputed on the modal choice instead of the single greedy read.

Reuses the committed stage-1 responses via --from-stage1 (stage 1 is not re-elicited) and the
production `run_experiment.run_stage2` (unmodified, default target-selection strategy) to build
exactly the R0-R4 conditions a real run would have built. The only new work here is stage 3's
loop, run --n-samples times per condition per requested temperature.

A hard, load-bearing limitation, not a bug: `harness.models.Backend.generate()` (both the vLLM
and the transformers implementation) hardcodes greedy decoding at `config.TEMPERATURE`/
`config.SEED` -- there is no per-call temperature or seed override in that interface. This
module does not, and must not, reimplement vLLM/transformers sampling to work around that (see
this module's ownership note); it can only run the *real* backend at `config.TEMPERATURE`
(0.0), a case that -- correctly -- produces zero sample-to-sample variation, which is reported,
not hidden (see `degenerate_zero_variance` in the per-temperature summary). Any temperature
other than `config.TEMPERATURE` against a real backend is refused outright, loudly, before any
GPU is touched -- see `_refuse_unsupported_temperatures`. Sampling at a genuinely different temperature needs a small
addition to `harness.models.Backend.generate()` (a `temperature`/`seed` parameter threaded into
`SamplingParams`/`generate()`) that is outside this file's ownership. `--dry-run` is unaffected:
`StubBackend` has no such hardware backing, so the dry run exercises the full sampling,
modal-choice and agreement logic against synthetic per-sample variation (see
`_SweepStubResponder`).

    python -m harness.decoding_sweep --dry-run --temperatures 0.0 0.7 --n-samples 3
    python -m harness.decoding_sweep --from-stage1 results/exp3c/stage1_<model>.jsonl \\
        --greedy-stage3-dir results/exp3c --temperatures 0.7 --n-samples 5 \\
        --out-dir results/decoding_sweep_t07
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

from . import config as C
from . import extract, prompts, repair
from .data import Item, build_items, load_records_from_file
from .models import Backend, ModelUnavailable, StubBackend, get_backend
from .rescore import rebuild_items
from .run_experiment import (
    CONDITION_LABELS,
    CONTINUOUS_CONTRASTS,
    _stage1_row,
    mcnemar,
    run_stage2,
)
from .run_pilot import _OPTION_LINE, _StubReplies, _slug, compute_max_len
from .watchdog import Watchdog, commit_gpu_seconds, cumulative_gpu_seconds

log = logging.getLogger("decoding_sweep")

# Measured from code/results/gpu_ledger.csv's committed stage1 rows: 36,637.8 GPU-seconds over
# 15,600 items across 12 runs (three model families on the 3090 Ti), one free-text generate()
# call per item -- the same call shape this module's sweep uses (no forced-choice probe; see
# the module docstring for why the probe is not part of this sweep).
SECONDS_PER_GENERATE_CALL = 2.35


def estimate_cost_s(n_items: int, n_conditions: int, n_samples: int, n_temperatures: int) -> float:
    """One free-text generate() call per item, per condition, per sample, per temperature.
    A documented, checkable function of the sweep's own shape -- see `SECONDS_PER_GENERATE_CALL`
    for where the per-call estimate comes from."""
    return n_items * n_conditions * n_samples * n_temperatures * SECONDS_PER_GENERATE_CALL


def check_budget(estimated_s: float, *, label: str) -> None:
    """Refuse to start if `estimated_s` would push the project past its 20 h GPU allowance."""
    spent = cumulative_gpu_seconds()
    remaining = C.PROJECT_GPU_BUDGET_S - spent
    if estimated_s > remaining:
        raise SystemExit(
            f"{label}: estimated cost {estimated_s:.0f}s exceeds the {remaining:.0f}s remaining "
            f"of the {C.PROJECT_GPU_BUDGET_S:.0f}s ({C.PROJECT_GPU_BUDGET_S / 3600:.1f}h) "
            f"project allowance ({spent:.0f}s already spent, from {C.GPU_LEDGER}). Reduce "
            f"--limit, --n-samples or --temperatures, or free up budget, before retrying."
        )
    log.info("%s: estimated cost %.0fs; %.0fs remaining of the %.0fs allowance (proceeding)",
              label, estimated_s, remaining, C.PROJECT_GPU_BUDGET_S)


def derive_sample_seed(
    base_seed: int, item_id: str, condition: str, temperature: float, sample_index: int,
) -> int:
    """A deterministic seed for one (item, condition, temperature, sample) draw.

    `hashlib`, not Python's built-in `hash()`, which is salted per-process (`PYTHONHASHSEED`)
    unless explicitly disabled -- a seed derived from it would not reproduce across runs, which
    defeats the point of recording `base_seed` in the summary. Distinct inputs (any of the five
    parts differing) always hash to a different seed; the same five parts always hash to the
    same one, so the whole sweep is exactly reproducible from `base_seed` alone.
    """
    key = f"{base_seed}:{item_id}:{condition}:{temperature}:{sample_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _refuse_unsupported_temperatures(temperatures: list[float], *, dry_run: bool) -> None:
    """Stop, loudly, before any GPU is touched, if a real (non-stub) run asks for a temperature
    `harness.models.Backend.generate()` cannot actually honour. See the module docstring."""
    if dry_run:
        return
    unsupported = sorted({t for t in temperatures if t != C.TEMPERATURE})
    if unsupported:
        raise SystemExit(
            f"decoding_sweep: temperature(s) {unsupported} requested against a real backend, "
            f"but harness.models.Backend.generate() hardcodes greedy decoding at "
            f"config.TEMPERATURE ({C.TEMPERATURE}) with no per-call override -- only "
            f"{C.TEMPERATURE} (greedy, zero sample variance -- see 'degenerate_zero_variance' "
            "in the per-temperature summary) can "
            "actually be sampled against the real model today. Genuine sampling needs a "
            "temperature/seed parameter added to Backend.generate() in harness/models.py, which "
            "this module does not own and must not reimplement. Use --dry-run to exercise the "
            "rest of this pipeline against StubBackend instead."
        )


# --------------------------------------------------------------------------- one sampled call


class _SweepStubResponder:
    """Canned replies for --dry-run, varied by an externally-set `seed` attribute.

    `StubBackend`'s responder signature (`Callable[[int, Chat], str]`) carries no seed of its
    own -- `_sample_generate` sets `.seed` immediately before each `backend.generate()` call, so
    the dry run gets real sample-to-sample variation to compute modal choice and agreement over,
    without `Backend.generate()` needing a parameter it does not have. Reuses `run_pilot`'s own
    option-line regex rather than re-deriving it.
    """

    def __init__(self) -> None:
        self.seed = 0

    def __call__(self, index: int, chat) -> str:
        user = next(m["content"] for m in reversed(list(chat)) if m["role"] == "user")
        options = _OPTION_LINE.findall(user)
        if not options:
            return "I cannot tell from these profiles."
        letter, _title = options[self.seed % len(options)]
        return f"The answer is {letter}."


def _sample_generate(
    backend: Backend, chat: list[dict], *, temperature: float, seed: int,
    stub_responder: _SweepStubResponder | None,
) -> str:
    """One sampled free-text response at `temperature`, seeded by `seed`.

    Stub backend: genuinely varies with `seed` via `stub_responder` (see
    `_SweepStubResponder`). Real backend at `config.TEMPERATURE`: `Backend.generate()`'s own
    built-in greedy call, unmodified -- deterministic regardless of `seed`, which is the
    correct behaviour for greedy decoding, not a bug in this adapter. Real backend at any other
    temperature: never reached -- `_refuse_unsupported_temperatures` stops the run before this
    function is ever called with one.
    """
    if backend.kind == "stub":
        if stub_responder is not None:
            stub_responder.seed = seed
        return backend.generate([chat])[0]
    if temperature == C.TEMPERATURE:
        return backend.generate([chat])[0]
    raise RuntimeError(
        f"_sample_generate: temperature {temperature} against a non-stub backend should have "
        "been refused before this call -- see _refuse_unsupported_temperatures"
    )


# --------------------------------------------------------------------------- the sweep itself


def _edited_letter_by_condition(item: Item, conditions: dict[str, Item], rival_letter: str) -> dict:
    """The option letter each condition edits -- R0 edits nothing, R1/R2 edit the rival, R3/R4
    edit whichever option `repair._edited_slot` finds changed. Mirrors
    `run_experiment.run_stage3`'s own derivation exactly (reused via the same helper, not
    re-derived by a second, possibly-diverging rule)."""
    r3_idx, _ = repair._edited_slot(item, conditions["R3"])
    r3_letter = chr(ord("A") + r3_idx) if r3_idx is not None else None
    return {"R0": None, "R1": rival_letter, "R2": rival_letter, "R3": r3_letter, "R4": r3_letter}


def run_stage3_sweep(
    backend: Backend,
    kept: dict[str, tuple],
    out_dir: Path,
    *,
    temperatures: list[float],
    n_samples: int,
    base_seed: int,
    greedy_by_item: dict[str, dict[str, str | None]] | None,
    stub_responder: "_SweepStubResponder | None" = None,
) -> tuple[dict, dict]:
    """Sample stage 3 `n_samples` times per item per condition per temperature.

    Returns (per-temperature summary dict, funnel counts). Writes one JSONL row per
    (item, condition, temperature, sample) with the full raw response kept, exactly the "never
    overwritten, raw output always kept" convention every other stage file in this project uses.

    `greedy_by_item` is the reference choice per (item, condition) to measure agreement against
    -- either loaded from a previously committed `stage3_<model>.jsonl` (see `--greedy-stage3-dir`)
    or, when `config.TEMPERATURE` is itself among `temperatures`, filled in from that
    temperature's own (deterministic, single-valued) sample as the sweep runs -- see `main`.

    `stub_responder` must be the *same* `_SweepStubResponder` instance `backend` itself was
    built with when `backend.kind == "stub"` -- mutating a second, freshly-constructed instance's
    `.seed` would never reach the one `StubBackend.generate()` actually calls, silently
    collapsing every sample to whatever the first draw happened to be (caught by
    `test_decoding_sweep.py::test_dry_run_produces_sample_to_sample_variation`, which is exactly
    why this is a required, explicit parameter rather than something this function constructs
    for itself).
    """
    label = f"{_slug(backend.name)}/decoding_sweep"
    path = out_dir / f"decoding_sweep_stage3_{_slug(backend.name)}.jsonl"
    total = len(kept) * len(CONDITION_LABELS) * len(temperatures) * n_samples
    dog = Watchdog(label=label, total=total, budget_min=C.PER_MODEL_BUDGET_MIN,
                    heartbeat=out_dir / "heartbeat.json")
    dog.sample_vram()

    if backend.kind == "stub" and stub_responder is None:
        raise ValueError("run_stage3_sweep: a stub backend requires its own stub_responder")
    computed_greedy: dict[str, dict[str, str | None]] = {}
    # choices[temperature][item_id][condition] = [choice_or_None, ...] (length n_samples)
    choices: dict[float, dict[str, dict[str, list]]] = {
        t: collections.defaultdict(dict) for t in temperatures
    }
    abort_reason = None

    try:
        with open(path, "a", encoding="utf-8") as fh:
            for item_id, (item, rejection, conditions) in kept.items():
                rival_letter = rejection.letter
                edited_letter = _edited_letter_by_condition(item, conditions, rival_letter)
                for temperature in temperatures:
                    for label_c in CONDITION_LABELS:
                        if dog.should_abort:
                            abort_reason = dog.abort_reason
                            break
                        cond_item = conditions[label_c]
                        chat = prompts.render(cond_item, "elicited")
                        samples: list[str | None] = []
                        for k in range(n_samples):
                            seed = derive_sample_seed(base_seed, item_id, label_c, temperature, k)
                            response = _sample_generate(
                                backend, chat, temperature=temperature, seed=seed,
                                stub_responder=stub_responder,
                            )
                            analysis = extract.analyse(response, cond_item)
                            samples.append(analysis.choice)
                            fh.write(json.dumps({
                                "model": backend.name, "item_id": item_id, "condition": label_c,
                                "temperature": temperature, "sample_index": k, "seed": seed,
                                "choice": analysis.choice, "edited_letter": edited_letter[label_c],
                                "response": response,
                            }, ensure_ascii=False) + "\n")
                            fh.flush()
                            dog.tick()
                        choices[temperature][item_id][label_c] = samples
                        if temperature == C.TEMPERATURE:
                            computed_greedy.setdefault(item_id, {})[label_c] = (
                                samples[0] if samples else None
                            )
                    if abort_reason:
                        break
                if abort_reason:
                    break
    finally:
        pass

    if backend.kind != "stub":
        commit_gpu_seconds(label=label, model=backend.name, backend=backend.kind,
                            items=len(kept) * len(temperatures), seconds=dog.elapsed_s,
                            vram_peak=dog.vram_peak, abort_reason=abort_reason)

    reference = greedy_by_item if greedy_by_item is not None else computed_greedy
    greedy_source = "loaded_from_file" if greedy_by_item is not None else "computed_from_sweep"

    per_temperature: dict[str, dict] = {}
    for temperature in temperatures:
        per_temperature[str(temperature)] = _summarise_temperature(
            choices[temperature], reference, temperature=temperature, n_samples=n_samples,
        )

    counts = {"n_items": len(kept), "n_temperatures": len(temperatures), "n_samples": n_samples,
              "abort_reason": abort_reason, "elapsed_min": round(dog.elapsed_s / 60.0, 2),
              "greedy_source": greedy_source}
    return per_temperature, counts


def _modal_choice(samples: list[str | None]) -> str | None:
    """The most frequent non-null choice, ties broken by first occurrence -- None only when
    every sample was unreadable."""
    present = [s for s in samples if s is not None]
    if not present:
        return None
    counts = collections.Counter(present)
    best = max(counts.values())
    for s in present:
        if counts[s] == best:
            return s
    return None  # unreachable, but no silent fallthrough


def _summarise_temperature(
    by_item: dict[str, dict[str, list]],
    reference: dict[str, dict[str, str | None]],
    *,
    temperature: float,
    n_samples: int,
) -> dict:
    """Per-condition choice distributions, modal choice per item, agreement with `reference`,
    and the four contrasts recomputed on the modal choice -- everything the module docstring
    promises, for one temperature."""
    distributions: dict[str, collections.Counter] = {c: collections.Counter() for c in CONDITION_LABELS}
    modal_by_item: dict[str, dict[str, str | None]] = {}
    modal_outcomes: dict[str, dict[str, bool | None]] = {}
    agreement = {c: {"n_with_reference": 0, "modal_matches": 0, "sample_matches": 0,
                      "sample_total": 0} for c in CONDITION_LABELS}

    for item_id, per_condition in by_item.items():
        modal_by_item[item_id] = {}
        modal_outcomes[item_id] = {}
        for cond, samples in per_condition.items():
            for s in samples:
                distributions[cond][s if s is not None else "<unreadable>"] += 1
            modal = _modal_choice(samples)
            modal_by_item[item_id][cond] = modal

            ref = reference.get(item_id, {}).get(cond)
            if ref is not None:
                agreement[cond]["n_with_reference"] += 1
                if modal is not None and modal == ref:
                    agreement[cond]["modal_matches"] += 1
                agreement[cond]["sample_total"] += len(samples)
                agreement[cond]["sample_matches"] += sum(1 for s in samples if s == ref)

    # The four contrasts on the modal choice are not computed here: they need each item's
    # *edited* letter (R3/R4's target varies per item), which this function is never handed --
    # only the raw sample lists. `main` computes them afterwards via `mcnemar_on_modal_choice`,
    # fed `modal_choice_by_item` below plus the edited-letter dict `_edited_letter_by_condition`
    # already derives from the same conditions stage 2 built.
    return {
        "distributions": {c: dict(v) for c, v in distributions.items()},
        "modal_choice_by_item": modal_by_item,
        "agreement_with_greedy": {
            c: {
                "n_items_with_reference": a["n_with_reference"],
                "modal_agreement_rate": (
                    a["modal_matches"] / a["n_with_reference"] if a["n_with_reference"] else None
                ),
                "sample_agreement_rate": (
                    a["sample_matches"] / a["sample_total"] if a["sample_total"] else None
                ),
            }
            for c, a in agreement.items()
        },
        "degenerate_zero_variance": temperature == C.TEMPERATURE,
        "n_samples": n_samples,
    }


def mcnemar_on_modal_choice(
    modal_by_item: dict[str, dict[str, str | None]],
    edited_letter_by_item: dict[str, dict[str, str | None]],
) -> dict:
    """The four `run_experiment.CONTINUOUS_CONTRASTS` pairs, McNemar-tested
    (`run_experiment.mcnemar`, reused unmodified) on "did the modal choice land on the option
    this condition edited" -- the discrete analogue of `chosen_is_edited`, generalised (unlike
    "chosen the rival") to R3/R4 whose edited option is not the rival at all.
    """
    outcomes: dict[str, dict[str, bool | None]] = {}
    for item_id, per_condition in modal_by_item.items():
        edited = edited_letter_by_item.get(item_id, {})
        outcomes[item_id] = {}
        for cond, modal in per_condition.items():
            target = edited.get(cond)
            outcomes[item_id][cond] = None if target is None or modal is None else modal == target
    return {f"{a}_vs_{b}": mcnemar(outcomes, a, b) for a, b in CONTINUOUS_CONTRASTS}


# --------------------------------------------------------------------------------- main


def _load_greedy_stage3(path: Path) -> dict[str, dict[str, str | None]]:
    """{item_id: {condition: choice}} read off a previously committed stage3_<model>.jsonl."""
    out: dict[str, dict[str, str | None]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.setdefault(row["item_id"], {})[row["condition"]] = row.get("choice")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E4: stage 3 re-sampled at one or more temperatures, N samples/item"
    )
    ap.add_argument("--temperatures", type=float, nargs="+", required=True,
                     help="one or more decoding temperatures to sample at -- required, no "
                          "default, because the cost scales directly with how many are given")
    ap.add_argument("--n-samples", type=int, required=True,
                     help="samples per item per condition per temperature -- required, no "
                          "default, for the same reason")
    ap.add_argument("--from-stage1", type=Path, default=None,
                     help="a stage1_*.jsonl from a previous run -- stage 1 is replayed, never "
                          "re-elicited. Required unless --dry-run.")
    ap.add_argument("--greedy-stage3-dir", type=Path, default=None,
                     help="a directory containing a previously committed stage3_<model>.jsonl "
                          "per model (e.g. results/exp3c), used as the greedy reference for the "
                          "agreement measure at no extra GPU cost. If omitted, config.TEMPERATURE "
                          "must be among --temperatures so the sweep computes its own reference.")
    ap.add_argument("--dry-run", action="store_true",
                     help="run the whole pipeline against a stub model on the bundled fixture: "
                          "no GPU, no network")
    ap.add_argument("--base-seed", type=int, default=C.SEED,
                     help="base seed for the per-(item,condition,temperature,sample) draw; "
                          "recorded in the summary so the sweep is reproducible from it alone")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap the number of items replayed per model")
    ap.add_argument("--models", nargs="*", default=None,
                     help="restrict to these model names among those present in --from-stage1")
    ap.add_argument("--part", default=None,
                     help="'I/N': process only the I-th of N deterministic shards of the item "
                          "set (1-based I)")
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--data-file", type=Path, default=None,
                     help="local JSON/JSONL dump instead of the hub, for rebuilding items "
                          "offline")
    ap.add_argument("--scan", type=int, default=0,
                     help="records to scan when rebuilding items from the hub (default: "
                          "40x the number of distinct items in --from-stage1)")
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR / "decoding_sweep")
    args = ap.parse_args(argv)

    if not args.dry_run and args.from_stage1 is None:
        ap.error("--from-stage1 is required unless --dry-run is set")
    if args.n_samples < 1:
        ap.error("--n-samples must be at least 1")
    if args.greedy_stage3_dir is None and C.TEMPERATURE not in args.temperatures:
        ap.error(f"need --greedy-stage3-dir <dir> (previously committed stage3_<model>.jsonl "
                  f"files) or {C.TEMPERATURE} among --temperatures, to have a greedy reference "
                  "to measure agreement against")
    _refuse_unsupported_temperatures(args.temperatures, dry_run=args.dry_run)

    protected_dir = (C.RESULTS_DIR / "exp").resolve()
    if args.out_dir.resolve() == protected_dir:
        ap.error(f"--out-dir must not be {protected_dir}: results/exp/ is a committed "
                  "scientific record and this run must write somewhere else")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(args.out_dir / "decoding_sweep.log")
    log.info("started %s on %s, temperatures=%s, n_samples=%d",
              C.stamp_utc(), platform.platform(), args.temperatures, args.n_samples)

    part = _parse_part(args.part)

    if args.dry_run:
        records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
        items = build_items(records, n_items=args.limit or 4, n_options=C.N_OPTIONS, seed=C.SEED)
        if not items:
            log.error("dry run: no usable items from the fixture")
            return 2
        # Stage-1 elicitation needs a genuine contrastive rejection to select an item at all --
        # `_StubReplies` (from `run_pilot`, the same stub `run_experiment --dry-run` uses)
        # crafts one; `_SweepStubResponder` below is deliberately bare (just a letter) and is
        # only for the sweep's own repeated stage-3 sampling, where the modal-choice logic is
        # what needs varying, not the stage-1 selection funnel.
        stub = StubBackend(_StubReplies(), name="stub")
        stage1_rows = {}
        for item in items:
            response = stub.generate([prompts.render(item, "elicited")])[0]
            stage1_rows[item.item_id] = _stage1_row("stub", "stub", item, response)
        model_names = ["stub"]
        per_model_items = {"stub": items}
        per_model_rows = {"stub": stage1_rows}
    else:
        raw = [json.loads(l) for l in args.from_stage1.read_text(encoding="utf-8").splitlines()
               if l.strip()]
        if not raw:
            log.error("no rows in %s", args.from_stage1)
            return 2
        from_stage1_rows: dict[str, list[dict]] = {}
        for row in raw:
            from_stage1_rows.setdefault(row["model"], []).append(row)
        model_names = sorted(from_stage1_rows)
        if args.models:
            model_names = [m for m in model_names if m in args.models]
            if not model_names:
                log.error("none of --models %s are present in %s", args.models, args.from_stage1)
                return 2
        log.info("replaying stage 1 from %s for model(s) %s", args.from_stage1, model_names)

        scan = args.scan or 40 * len({r["item_id"] for r in raw})
        per_model_items = {}
        per_model_rows = {}
        for name in model_names:
            rows = from_stage1_rows[name]
            wanted_ids = _select_part(sorted({r["item_id"] for r in rows}), part)
            if args.limit is not None:
                wanted_ids = wanted_ids[:args.limit]
            rebuilt = rebuild_items(rows, scan, args.data_file)
            per_model_items[name] = [rebuilt[i] for i in wanted_ids]
            per_model_rows[name] = {r["item_id"]: r for r in rows if r["item_id"] in wanted_ids}

    all_items = [i for its in per_model_items.values() for i in its]
    corpus_index = repair.build_corpus_index(all_items)

    # Stage 2 (free, no GPU) for every requested model first, so the budget check below sees
    # the true item count before anything expensive happens.
    stage2_by_model: dict[str, tuple[dict, dict]] = {}
    total_kept = 0
    for name in model_names:
        kept, stage2_counts = run_stage2(per_model_items[name], per_model_rows[name],
                                          corpus_index, args.out_dir, name)
        stage2_by_model[name] = (kept, stage2_counts)
        total_kept += len(kept)
        log.info("%s stage2: %s", name,
                  json.dumps({k: v for k, v in stage2_counts.items() if k != "drop_reasons"}))

    estimated = estimate_cost_s(total_kept, len(CONDITION_LABELS), args.n_samples,
                                 len(args.temperatures))
    check_budget(estimated, label="decoding_sweep")

    max_len = compute_max_len(all_items) if all_items else C.MAX_MODEL_LEN_FLOOR

    summary: dict[str, dict] = {}
    for name in model_names:
        kept, stage2_counts = stage2_by_model[name]
        model_summary = {"stage2": stage2_counts}
        if not kept:
            log.warning("%s: nothing survived stage 2, skipping the sweep", name)
            summary[name] = model_summary
            continue

        stub_responder = None
        if args.dry_run:
            # The same instance goes into the backend and into run_stage3_sweep below -- see
            # run_stage3_sweep's docstring for why a second, freshly-constructed instance would
            # silently break sample variation.
            stub_responder = _SweepStubResponder()
            backend = StubBackend(stub_responder, name="stub")
        else:
            try:
                backend = get_backend(name, kind=args.backend, max_model_len=max_len,
                                       allow_mirror=not args.no_mirror)
            except ModelUnavailable as exc:
                log.error("%s could not be loaded: %s", name, exc)
                summary[name] = model_summary
                continue

        greedy_by_item = None
        if args.greedy_stage3_dir is not None:
            greedy_path = args.greedy_stage3_dir / f"stage3_{_slug(backend.name)}.jsonl"
            if greedy_path.exists():
                greedy_by_item = _load_greedy_stage3(greedy_path)
            elif C.TEMPERATURE not in args.temperatures:
                log.error("%s: no committed greedy result at %s and %s is not among "
                          "--temperatures; skipping", name, greedy_path, C.TEMPERATURE)
                summary[name] = model_summary
                if backend.kind != "stub":
                    backend.close()
                continue

        per_temperature, sweep_counts = run_stage3_sweep(
            backend, kept, args.out_dir, temperatures=args.temperatures,
            n_samples=args.n_samples, base_seed=args.base_seed, greedy_by_item=greedy_by_item,
            stub_responder=stub_responder,
        )
        log.info("%s sweep: %s", backend.name, json.dumps(sweep_counts))
        model_summary["sweep"] = sweep_counts
        model_summary["per_temperature"] = per_temperature

        # The contrasts, recomputed on the modal choice, for every temperature -- edited_letter
        # is re-derived per item (cheap, pure text) from the same conditions stage 2 built.
        edited_letter_by_item = {
            item_id: _edited_letter_by_condition(item, conditions, rejection.letter)
            for item_id, (item, rejection, conditions) in kept.items()
        }
        model_summary["mcnemar_on_modal_choice"] = {
            temp_key: mcnemar_on_modal_choice(block["modal_choice_by_item"], edited_letter_by_item)
            for temp_key, block in per_temperature.items()
        }

        summary[name] = model_summary
        if backend.kind != "stub":
            backend.close()

    payload = {
        "finished_utc": C.stamp_utc(),
        "experiment": "decoding_sweep (E4)",
        "temperatures": args.temperatures,
        "n_samples": args.n_samples,
        "base_seed": args.base_seed,
        "from_stage1": str(args.from_stage1) if not args.dry_run else None,
        "greedy_stage3_dir": str(args.greedy_stage3_dir) if args.greedy_stage3_dir else None,
        "part": args.part,
        "model_list": model_names,
        "n_items_kept_total": total_kept,
        "estimated_cost_s": round(estimated, 1),
        "dry_run": args.dry_run,
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "per_model": summary,
    }
    out_path = args.out_dir / "decoding_sweep_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


def _parse_part(spec: str | None) -> tuple[int, int] | None:
    """"--part I/N" -> (0-based index, N), or None. See `gate8_variant._parse_part`, which this
    duplicates deliberately -- each of these two scripts is meant to be runnable and readable
    standalone, and this is five lines, not a shared abstraction worth a new import edge."""
    if spec is None:
        return None
    try:
        i_str, n_str = spec.split("/", 1)
        i, n = int(i_str), int(n_str)
    except ValueError:
        raise SystemExit(f"--part must look like 'I/N' (e.g. '1/3'), got {spec!r}") from None
    if n < 1 or not (1 <= i <= n):
        raise SystemExit(f"--part {spec!r}: need 1 <= I <= N, got I={i}, N={n}")
    return i - 1, n


def _select_part(item_ids: list[str], part: tuple[int, int] | None) -> list[str]:
    """Every `n`-th id (stable order), starting at `index`. See `gate8_variant._select_part`."""
    if part is None:
        return item_ids
    index, n = part
    return [iid for pos, iid in enumerate(sorted(item_ids)) if pos % n == index]


if __name__ == "__main__":
    sys.exit(main())
