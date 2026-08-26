"""E3: does the R1-R3 / R2-R4 *location* effect survive when R3/R4's target is chosen without
regard to whether it already lacks the named attribute?

Gate 8 (see `pilot.repair.check_integrity`) requires R3/R4's target to lack the named attribute
before the edit, and `build_conditions_with_diagnostics`'s default target-selection rule
("prefer_lacking") *prefers* a candidate with that property. That coupling means a location
effect measured under the default rule cannot tell apart two explanations: the model's
rejection is not specific to the option it named (a genuine location effect), or the rejection
named a criterion that happens to describe several options at once (an artefact of always
building R3/R4 out of options that already satisfy it). Choosing the target uniformly at random,
ignoring the attribute split, breaks that coupling: it still builds R0-R4 exactly as
`run_experiment` does otherwise (same rival, same R1/R2 source search, same eight gates), but
the option R3/R4 land on no longer has anything to do with what the model complained about.

Three strategies are wired through `repair.select_r3_target_index` (imported, not
reimplemented): "prefer_lacking" (today's default, reproduced here for a same-code comparison),
"random" (the arm this module exists for), and "prefer_having" (the true counterfactual -- a
target that already carries the attribute, which fails gate 8 by construction and is expected to
build few or no items; included so it is at least expressible and its funnel is visible, not
because it is expected to produce a usable run).

Reuses the committed stage-1 responses via --from-stage1, exactly like
`run_experiment --from-stage1`: stage 1 is not re-elicited, so no stage-1 GPU cost is repaid.
Stage 2 (building R0-R4 and running the eight integrity gates) is pure text and costs nothing.
Only stage 3 -- re-asking under the five conditions -- touches the GPU, and it is the same
two-call-per-condition shape `run_experiment.run_stage3` already uses (reused here, not
reimplemented).

    python -m pilot.gate8_variant --dry-run --strategy random --limit 4
    python -m pilot.gate8_variant --from-stage1 results/exp3c/stage1_<model>.jsonl \\
        --strategy random --out-dir results/gate8_variant_random

Three stages per model, same split as `run_experiment`:

    stage 1  replayed from --from-stage1, never re-run.
    stage 2  build R0-R4 under the requested strategy and run the eight integrity gates; per
             item, records which strategy chose the target, which letter it chose, how many
             candidates were available, and how many of those lacked the attribute -- so a
             "prefer_lacking" run and a "random" run can be compared item by item.
    stage 3  re-ask under R0-R4 (`run_experiment.run_stage3`, unchanged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import sys
from pathlib import Path

from . import config as C
from . import prompts, repair
from .data import Item, build_items, load_records_from_file
from .models import ModelUnavailable, StubBackend, get_backend
from .rescore import rebuild_items
from .run_experiment import (
    _drop_category,
    _selected_rejection,
    _stage1_row,
    all_continuous_contrasts,
    mcnemar,
    r0_baseline_diagnostic,
    run_stage3,
)
from .run_pilot import _StubReplies, _slug, compute_max_len
from .watchdog import cumulative_gpu_seconds

log = logging.getLogger("gate8_variant")

# Measured from code/results/gpu_ledger.csv's committed stage3 rows: 22,796.6 GPU-seconds over
# 1,857 items across 17 runs (three model families on the 3090 Ti) -- five conditions, a
# free-text call plus a forced-choice probe call each, the exact shape `run_stage3` uses. This
# is a per-item estimate, not a per-call one, because that is the unit `run_stage3` itself is
# billed in.
SECONDS_PER_STAGE3_ITEM = 12.3


def estimate_cost_s(n_items: int) -> float:
    """Estimated GPU-seconds for stage 3 over `n_items` gate-clean items, one model.

    A documented function of item count alone (see `SECONDS_PER_STAGE3_ITEM`), so a caller can
    read and check the estimate before spending anything -- stage 1 is never re-run here (see
    the module docstring) and stage 2 is free, so stage 3 is the whole cost.
    """
    return n_items * SECONDS_PER_STAGE3_ITEM


def check_budget(estimated_s: float, *, label: str) -> None:
    """Refuse to start if `estimated_s` would push the project past its 20 h GPU allowance.

    Raises `SystemExit` with a clear message rather than returning False, because the caller
    has nothing sensible to do but stop -- this is meant to run once, before any backend loads.
    """
    spent = cumulative_gpu_seconds()
    remaining = C.PROJECT_GPU_BUDGET_S - spent
    if estimated_s > remaining:
        raise SystemExit(
            f"{label}: estimated cost {estimated_s:.0f}s exceeds the {remaining:.0f}s remaining "
            f"of the {C.PROJECT_GPU_BUDGET_S:.0f}s ({C.PROJECT_GPU_BUDGET_S / 3600:.1f}h) "
            f"project allowance ({spent:.0f}s already spent, from {C.GPU_LEDGER}). Reduce "
            f"--limit or --part, or free up budget, before retrying."
        )
    log.info("%s: estimated cost %.0fs; %.0fs remaining of the %.0fs allowance (proceeding)",
              label, estimated_s, remaining, C.PROJECT_GPU_BUDGET_S)


def _item_rng(seed: int, item_id: str) -> random.Random:
    """A `random.Random` seeded deterministically from `seed` and `item_id`.

    Uses `hashlib` rather than Python's built-in `hash()`, which is salted per-process
    (`PYTHONHASHSEED`) unless explicitly disabled -- a seed derived from it would not reproduce
    across runs, defeating the whole point of recording a seed. sha256 truncated to 8 bytes is
    plenty of entropy for a `random.Random` seed and is stable forever.
    """
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _parse_part(spec: str | None) -> tuple[int, int] | None:
    """"--part I/N" -> (0-based index, N), or None when no sharding was requested.

    Splitting a run into independent shards (each a separate process, each writing its own
    output files under a different --out-dir) is how a long sweep gets done within a wall-clock
    budget without one process holding the GPU the whole time. Validated eagerly so a typo is
    an argument error, not a silently-empty shard.
    """
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
    """Every `n`-th id (stable order), starting at `index` -- deterministic given `item_ids`'
    own order, so the same --part always selects the same items regardless of how many other
    shards exist or run."""
    if part is None:
        return item_ids
    index, n = part
    return [iid for pos, iid in enumerate(sorted(item_ids)) if pos % n == index]


# --------------------------------------------------------------------------------- stage 2


def run_stage2_variant(
    items: list[Item], stage1_rows: dict[str, dict], corpus_index: dict, out_dir: Path,
    model: str, *, strategy: str, seed: int,
) -> tuple[dict[str, tuple[Item, repair.Rejection, dict]], dict]:
    """`run_experiment.run_stage2`'s counterpart under a named R3/R4 target-selection strategy.

    Not a call to `run_experiment.run_stage2` with an extra argument -- that function has no
    strategy parameter, deliberately (see run_experiment.py's ownership note); this is the
    thinnest wrapper that gets the same behaviour for "prefer_lacking" and adds the other two
    strategies, reusing `repair.build_conditions_with_diagnostics` exactly as
    `run_experiment.run_stage2` does. Writes its own JSONL, `gate8_<strategy>_stage2_<model>.jsonl`,
    rather than `stage2_<model>.jsonl`, so a "prefer_lacking" replay run here is never confused
    with -- or overwrites -- the original run's own stage-2 record.
    """
    path = out_dir / f"gate8_{strategy}_stage2_{_slug(model)}.jsonl"
    kept: dict[str, tuple[Item, repair.Rejection, dict]] = {}
    drop_reasons: dict[str, int] = {}
    counts = {"attempted": 0, "built": 0, "dropped": 0, "search_cap_hit": 0, "strategy": strategy}

    with open(path, "a", encoding="utf-8") as fh:
        for item in items:
            row = stage1_rows.get(item.item_id)
            if row is None:
                continue
            rejection = _selected_rejection(item, row)
            if rejection is None:
                continue
            counts["attempted"] += 1

            rng = _item_rng(seed, item.item_id) if strategy == "random" else None
            record = {"model": model, "item_id": item.item_id, "rival_title": rejection.title,
                      "attribute": rejection.attribute, "target_strategy": strategy}
            try:
                conditions, diag = repair.build_conditions_with_diagnostics(
                    item, rejection, corpus_index, row["choice"],
                    r3_strategy=strategy, r3_rng=rng,
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
            # The per-item comparison record the module docstring promises: which strategy
            # chose the target, which letter it chose, how many candidates were available, and
            # how many of those lacked the attribute -- everything `RepairDiagnostics` already
            # carries, plus the strategy name it does not (see `RepairDiagnostics.r3_strategy`,
            # which this agrees with by construction: both come from the same `strategy` value).
            record.update(built=True, drop_reason=None, drop_category=None, stratum=diag.stratum,
                          r1_examined=diag.r1_examined, r2_examined=diag.r2_examined,
                          r1_total=diag.r1_total, r2_total=diag.r2_total, cap_hit=diag.cap_hit,
                          r3_candidates_total=diag.r3_candidates_total,
                          r3_candidates_without_attribute=diag.r3_candidates_without_attribute,
                          r3_picked_letter=diag.r3_picked_letter,
                          r3_strategy_recorded=diag.r3_strategy)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept[item.item_id] = (item, rejection, conditions)

    counts["drop_reasons"] = drop_reasons
    return kept, counts


# --------------------------------------------------------------------------------- main


def _source_name(args) -> str:
    if args.data_file:
        return str(args.data_file)
    if args.dry_run:
        return "fixture:twowiki_sample.jsonl"
    return f"stage1 replay: {args.from_stage1}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E3: R3/R4 target selection under a named strategy, replayed from a "
                     "committed stage-1 file"
    )
    ap.add_argument("--strategy", required=True, choices=repair.R3_STRATEGIES,
                     help="which rule chooses R3/R4's target -- 'random' is the arm this "
                          "module exists for; 'prefer_lacking' reproduces the default rule for "
                          "a same-code comparison; 'prefer_having' is the true counterfactual "
                          "and is expected to build few or no items (see the module docstring)")
    ap.add_argument("--from-stage1", type=Path, default=None,
                     help="a stage1_*.jsonl from a previous run -- stage 1 is replayed, never "
                          "re-elicited, so no stage-1 GPU cost is repaid. Required unless "
                          "--dry-run.")
    ap.add_argument("--dry-run", action="store_true",
                     help="run the whole pipeline against a stub model on the bundled fixture: "
                          "no GPU, no network. --from-stage1 is not needed (and is ignored if "
                          "given).")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap the number of items replayed per model (default: every item in "
                          "--from-stage1)")
    ap.add_argument("--models", nargs="*", default=None,
                     help="restrict to these model names among those present in --from-stage1 "
                          "(default: every model present)")
    ap.add_argument("--part", default=None,
                     help="'I/N': process only the I-th of N deterministic shards of the item "
                          "set (1-based I), so a run can be split across processes -- see "
                          "_select_part")
    ap.add_argument("--seed", type=int, default=C.SEED,
                     help="base seed for the 'random' strategy's per-item draw; recorded in "
                          "the summary so the run is reproducible from it alone")
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--data-file", type=Path, default=None,
                     help="local JSON/JSONL dump instead of the hub, for rebuilding items "
                          "offline")
    ap.add_argument("--scan", type=int, default=0,
                     help="records to scan when rebuilding items from the hub (default: "
                          "40x the number of distinct items in --from-stage1)")
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR / "gate8_variant")
    args = ap.parse_args(argv)

    if not args.dry_run and args.from_stage1 is None:
        ap.error("--from-stage1 is required unless --dry-run is set")

    protected_dir = (C.RESULTS_DIR / "exp").resolve()
    if args.out_dir.resolve() == protected_dir:
        ap.error(f"--out-dir must not be {protected_dir}: results/exp/ is a committed "
                  "scientific record and this run must write somewhere else")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(args.out_dir / "gate8_variant.log")
    log.info("started %s on %s, strategy=%s", C.stamp_utc(), platform.platform(), args.strategy)

    part = _parse_part(args.part)

    if args.dry_run:
        records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
        items = build_items(records, n_items=args.limit or 4, n_options=C.N_OPTIONS, seed=C.SEED)
        if not items:
            log.error("dry run: no usable items from the fixture")
            return 2
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
        rebuild_data_file = args.data_file
        per_model_items = {}
        per_model_rows = {}
        for name in model_names:
            rows = from_stage1_rows[name]
            wanted_ids = _select_part(sorted({r["item_id"] for r in rows}), part)
            if args.limit is not None:
                wanted_ids = wanted_ids[:args.limit]
            rebuilt = rebuild_items(rows, scan, rebuild_data_file)
            per_model_items[name] = [rebuilt[i] for i in wanted_ids]
            per_model_rows[name] = {r["item_id"]: r for r in rows if r["item_id"] in wanted_ids}

    # One corpus index shared across models, exactly as run_experiment builds one from the full
    # item set (siblings across models are the same underlying items, rebuilt once per model
    # above only because rebuild_items is keyed by that model's stage-1 rows).
    all_items = [i for items in per_model_items.values() for i in items]
    corpus_index = repair.build_corpus_index(all_items)

    # Budget check happens once, before any backend loads, using the built (free, no-GPU)
    # stage-2 item counts across every requested model -- see check_budget / estimate_cost_s.
    stage2_by_model: dict[str, tuple[dict, dict]] = {}
    total_kept = 0
    for name in model_names:
        kept, stage2_counts = run_stage2_variant(
            per_model_items[name], per_model_rows[name], corpus_index, args.out_dir, name,
            strategy=args.strategy, seed=args.seed,
        )
        stage2_by_model[name] = (kept, stage2_counts)
        total_kept += len(kept)
        log.info("%s stage2 (%s): %s", name, args.strategy,
                  json.dumps({k: v for k, v in stage2_counts.items() if k != "drop_reasons"}))

    estimated = estimate_cost_s(total_kept)
    check_budget(estimated, label="gate8_variant")

    max_len = compute_max_len(all_items) if all_items else C.MAX_MODEL_LEN_FLOOR

    summary: dict[str, dict] = {}
    all_drop_reasons: dict[str, int] = {}
    all_prob_outcomes: dict[str, dict[str, dict]] = {}

    for name in model_names:
        kept, stage2_counts = stage2_by_model[name]
        model_summary = {"stage2": stage2_counts}
        for reason, n in stage2_counts["drop_reasons"].items():
            all_drop_reasons[reason] = all_drop_reasons.get(reason, 0) + n

        if not kept:
            log.warning("%s: nothing survived stage 2 under %s, skipping stage 3",
                        name, args.strategy)
            summary[name] = model_summary
            continue

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
        prob_outcomes = stage3_counts.pop("prob_outcomes", {})
        log.info("%s stage3: %s", backend.name, json.dumps(stage3_counts))
        model_summary["stage3"] = stage3_counts
        model_summary["mcnemar_r1_vs_r2"] = mcnemar(outcomes, "R1", "R2")
        model_summary["mcnemar_r1_vs_r3"] = mcnemar(outcomes, "R1", "R3")
        model_summary["continuous"] = all_continuous_contrasts(prob_outcomes, seed=args.seed)
        model_summary["r0_baseline_by_target"] = r0_baseline_diagnostic(prob_outcomes)
        for item_id, per_item_probs in prob_outcomes.items():
            all_prob_outcomes[f"{backend.name}::{item_id}"] = per_item_probs

        summary[backend.name] = model_summary
        if backend.kind != "stub":
            backend.close()

    payload = {
        "finished_utc": C.stamp_utc(),
        "experiment": "gate8_variant (E3)",
        "strategy": args.strategy,
        "dataset": _source_name(args),
        "from_stage1": str(args.from_stage1) if not args.dry_run else None,
        "seed": args.seed,
        "part": args.part,
        "model_list": model_names,
        "n_items_kept_total": total_kept,
        "estimated_cost_s": round(estimated, 1),
        "dry_run": args.dry_run,
        "drop_reasons": all_drop_reasons,
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "continuous_pooled": all_continuous_contrasts(all_prob_outcomes, seed=args.seed),
        "r0_baseline_by_target_pooled": r0_baseline_diagnostic(all_prob_outcomes),
        "per_model": summary,
    }
    out_path = args.out_dir / f"gate8_variant_{args.strategy}_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
