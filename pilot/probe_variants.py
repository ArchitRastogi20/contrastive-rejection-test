"""E2: does restructuring the forced-choice letter probe raise how often it completes, and does
it change the read it produces when it does?

The continuous outcome measure (`pilot.models.append_letter_probe` + `letter_probs`) appends one
instruction to the existing prompt and forces a single generated token, then reads that token's
probability mass on each candidate letter. That read is complete on every row for most models,
but is markedly less so for others -- on one model's rows it completes only a minority of the
time, and on a reasoning model it never completes at all, because that model's replies always
open with a `<think>` block before anything else, so the forced single token is never the
letter. This experiment tests whether a *different* probe shape does better on the same rows,
never by asking a model to grade another model -- every field here is either the model's own
next-token distribution or a deterministic function of it.

**The variants.** `VARIANTS` is a name -> `ProbeVariant` table, each entry a pure function
`Chat -> Chat` that builds the probe chat from the item's already-rendered "elicited" chat.
`"baseline"` is today's probe (`append_letter_probe`, unchanged) kept as the first arm so every
comparison is against the status quo, not just between two untested ideas. The other variants
prefill the assistant turn with a short stem and read the *next* token after it -- forcing the
model's reply to begin with an answer rather than merely asking it to. The table is deliberately
data, not a chain of `if variant == ...:` branches, so a fourth variant is one more entry.

**What a prefilled assistant turn actually does is not assumed.** Continuing an assistant turn
(`continue_final_message=True` on `apply_chat_template`, wired into `pilot.models.
render_for_probe`) depends on the tokenizer's chat template treating that flag as "keep going
from here" rather than silently reopening a fresh assistant header after the stem -- template
support for this is not uniform, and neither is whether it suppresses a reasoning model's
`<think>` preamble specifically (a template could honour the continuation and still let the
model open a new think block right after the stem, since nothing about continuation forbids
that). Both are exactly what this experiment measures: `continuation_confirmed` in every read's
`detail` records whether the template kept the stem intact, and completion-by-model is reported
per variant so a reasoning model's actual behaviour under a prefilled turn is data, not an
assumption made in this docstring.

**Read reuse, not reimplementation.** Every variant's read is built with the same
`append_letter_probe`, `renormalize_letter_logprobs`, `letter_read_from_token_logprobs` and
`resolve_letter_token_ids` the existing probe already uses (via `Backend.letter_probs`, which
this module never bypasses) -- only the chat a variant hands to that machinery differs.

**Reads already-built items, regenerates nothing upstream.** Same reconstruction as
`pilot.surprisal` (`pilot.surprisal.reconstruct_conditions`): the five R0-R4 conditions per item
were already built and already shown to the model in a committed run-3 stage-3 pass. This module
adds one new kind of model call (a forced-choice read under each variant's chat) on top of that
already-committed item set; it does not re-elicit or re-repair anything.

    python -m pilot.probe_variants --dry-run                    # whole pipeline, stub, no GPU
    python -m pilot.probe_variants --part B --models DeepSeek    # the real thing
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config as C
from . import prompts
from .models import (
    Backend,
    Chat,
    LetterProbRead,
    ModelUnavailable,
    StubBackend,
    append_letter_probe,
    get_backend,
    letter_read_from_token_logprobs,
    renormalize_letter_logprobs,
    resolve_letter_token_ids,
)
from .run_pilot import _StubReplies, _slug
from .surprisal import PARTS, ReconstructedItem, reconstruct_conditions
from .watchdog import Watchdog, commit_gpu_seconds, cumulative_gpu_seconds

log = logging.getLogger("probe_variants")

CONDITIONS = ("R0", "R1", "R2", "R3", "R4")

# --------------------------------------------------------------------------- the variant table


@dataclass(frozen=True)
class ProbeVariant:
    name: str
    description: str
    build: Callable[[Chat], list[dict]]


def _baseline(chat: Chat) -> list[dict]:
    return append_letter_probe(chat)


# Short, declarative stems: enough to force the very next token to open the answer, nothing a
# reasoning model would need to reason about first. Left as plain module constants (not folded
# into the lambdas below) so a reader -- or a fourth variant -- can see exactly what text every
# variant prefills without reading a closure body.
PREFILL_STEM = "Answer: "
PREFILL_PAREN_STEM = "The correct candidate is ("


def _prefill_stem(chat: Chat) -> list[dict]:
    """Same instruction as the baseline, plus a prefilled assistant turn so the forced token is
    the one immediately after a bare stem rather than the first token of a fresh reply."""
    return [*append_letter_probe(chat), {"role": "assistant", "content": PREFILL_STEM}]


def _prefill_paren(chat: Chat) -> list[dict]:
    """As `_prefill_stem`, but the stem ends on an open parenthesis -- a shape closer to how the
    free-text arm's own replies tend to open ("B) Candidate Name"), on the theory that a stem
    matching the model's own habitual phrasing is more likely to be honoured by the chat
    template as a continuation rather than read as something to react to."""
    return [*append_letter_probe(chat), {"role": "assistant", "content": PREFILL_PAREN_STEM}]


VARIANTS: dict[str, ProbeVariant] = {
    "baseline": ProbeVariant(
        "baseline",
        "today's probe: append the instruction to the existing user turn, "
        "then force one generated token",
        _baseline,
    ),
    "prefill_stem": ProbeVariant(
        "prefill_stem",
        f"prefill the assistant turn with {PREFILL_STEM!r} and force the next token",
        _prefill_stem,
    ),
    "prefill_paren": ProbeVariant(
        "prefill_paren",
        f"prefill the assistant turn with {PREFILL_PAREN_STEM!r} and force the next token",
        _prefill_paren,
    ),
}


def top_letter(read: LetterProbRead) -> str | None:
    """The argmax candidate, or None when nothing was read at all -- ties broken by candidate
    order (`probs` is built in that order), never at random."""
    if not read.probs:
        return None
    return max(read.probs, key=lambda letter: read.probs[letter])


# --------------------------------------------------------------------------- GPU budget

CALLS_PER_ROW = len(VARIANTS)  # one letter_probs call per variant, same (item, condition)

# Documented, unverified placeholder -- see `pilot.surprisal.EST_SECONDS_PER_CALL` for the same
# caveat. A `letter_probs` call is a single forced-token generation, the same shape run 3's own
# continuous-measure probe already used at roughly 2.6s/cell including a much longer free-text
# generation alongside it -- 0.5s/call is a conservative guess for the probe call alone, not a
# measurement. Replace with what the first real run's `gpu_ledger.csv` shows.
EST_SECONDS_PER_CALL = 0.5


def estimate_gpu_seconds(n_rows: int) -> float:
    """`n_rows` (item x condition x model) x `CALLS_PER_ROW` (one call per variant) x
    `EST_SECONDS_PER_CALL`."""
    return n_rows * CALLS_PER_ROW * EST_SECONDS_PER_CALL


def budget_check(
    n_rows: int, *, spent: float, allowance: float = C.PROJECT_GPU_BUDGET_S,
) -> dict:
    """Same shape and the same purity guarantee as `pilot.surprisal.budget_check`: no ledger
    I/O, so it is testable with a hand-supplied `spent`."""
    estimate = estimate_gpu_seconds(n_rows)
    remaining = allowance - spent
    return {
        "n_rows": n_rows, "estimate_s": estimate, "spent_s": spent, "allowance_s": allowance,
        "remaining_s": remaining, "ok": estimate <= remaining,
    }


# ------------------------------------------------------------------------------- the run itself


def run_probe_variants(
    backend: Backend, kept: dict[str, ReconstructedItem], out_dir: Path, part: str,
) -> tuple[list[dict], dict]:
    """Every variant's read, for every condition, for every reconstructed item. Writes
    probe_variants_<model>.jsonl and returns the rows plus completion/agreement counts."""
    label = f"{_slug(backend.name)}/probe_variants"
    path = out_dir / f"probe_variants_{_slug(backend.name)}.jsonl"
    total = len(kept) * len(CONDITIONS) * len(VARIANTS)
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
                for cond in CONDITIONS:
                    if dog.should_abort:
                        abort_reason = dog.abort_reason
                        break
                    cond_item = recon.conditions[cond]
                    chat = prompts.render(cond_item, "elicited")
                    candidates = [chr(ord("A") + i) for i in range(len(cond_item.options))]

                    for variant_name, variant in VARIANTS.items():
                        probe_chat = variant.build(chat)
                        read = backend.letter_probs([probe_chat], [candidates])[0]
                        row = {
                            "model": backend.name, "part": part, "item_id": item_id,
                            "condition": cond, "variant": variant_name, "candidates": candidates,
                            "complete": read.complete, "top_letter": top_letter(read),
                            "probs": read.probs, "raw_logprobs": read.raw_logprobs,
                            "detail": read.detail,
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
        "completion_by_variant": completion_by(rows_out, "variant"),
        "completion_by_condition": completion_by_two(rows_out, "variant", "condition"),
        "agreement": variant_agreement(rows_out),
    }
    return rows_out, counts


# ------------------------------------------------------------------------------ the statistics


def completion_by(rows: list[dict], key: str) -> dict:
    """{value-of-key: {"n":, "complete":, "rate":}}, e.g. completion per variant."""
    out: dict[str, dict] = {}
    for row in rows:
        bucket = out.setdefault(row[key], {"n": 0, "complete": 0})
        bucket["n"] += 1
        bucket["complete"] += int(row["complete"])
    for bucket in out.values():
        bucket["rate"] = bucket["complete"] / bucket["n"] if bucket["n"] else None
    return out


def completion_by_two(rows: list[dict], outer_key: str, inner_key: str) -> dict:
    """{value-of-outer: {value-of-inner: {"n":, "complete":, "rate":}}}, e.g. completion per
    variant per condition -- the finer-grained table the top-level request asks for alongside
    the flat per-variant one."""
    out: dict[str, dict] = {}
    for row in rows:
        outer = out.setdefault(row[outer_key], {})
        bucket = outer.setdefault(row[inner_key], {"n": 0, "complete": 0})
        bucket["n"] += 1
        bucket["complete"] += int(row["complete"])
    for outer in out.values():
        for bucket in outer.values():
            bucket["rate"] = bucket["complete"] / bucket["n"] if bucket["n"] else None
    return out


def variant_agreement(rows: list[dict]) -> dict:
    """For every pair of variants, over the rows (same item, condition, model) where *both*
    completed: how often they land on the same top letter.

    Paired, not marginal -- a variant that raises completion by picking up rows the other
    variant could never read at all must not silently deflate this by counting those rows as
    disagreements; they are excluded from the denominator here exactly as an incomplete read is
    excluded from every other paired measure in this project, and reported separately as the
    completion tables above.
    """
    by_key: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["model"], row["item_id"], row["condition"])
        by_key.setdefault(key, {})[row["variant"]] = row

    names = list(VARIANTS)
    out: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both_complete = 0
            agree = 0
            for per_variant in by_key.values():
                ra, rb = per_variant.get(a), per_variant.get(b)
                if ra is None or rb is None or not ra["complete"] or not rb["complete"]:
                    continue
                both_complete += 1
                if ra["top_letter"] == rb["top_letter"]:
                    agree += 1
            out[f"{a}_vs_{b}"] = {
                "n_both_complete": both_complete, "n_agree": agree,
                "rate": agree / both_complete if both_complete else None,
            }
    return out


# --------------------------------------------------------------------------- dry run, no GPU


def _dry_run_reconstruction() -> dict[str, ReconstructedItem]:
    """Same fixture-only construction `pilot.surprisal._dry_run_reconstruction` uses -- kept as
    its own copy (not imported) so this module's dry run does not depend on the internal
    surprisal-specific chat rendering `run_surprisal` performs, only on the item/condition
    reconstruction this module also needs."""
    from . import extract, repair
    from .data import build_items, load_records_from_file

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
    rows, counts = run_probe_variants(backend, kept, args.out_dir, None)
    log.info("dry-run probe_variants: %s",
              json.dumps({k: v for k, v in counts.items()
                          if k not in ("completion_by_variant", "completion_by_condition",
                                       "agreement")}))

    summary = {
        "finished_utc": C.stamp_utc(), "dry_run": True, "part": None, "seed": C.SEED,
        "models": [backend.name], "n_items": len(kept),
        "variants": {name: v.description for name, v in VARIANTS.items()},
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "per_model": {backend.name: counts},
    }
    out_path = args.out_dir / "probe_variants_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


# --------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E2: does a restructured letter probe complete more often, and read the "
                     "same thing when it does?"
    )
    ap.add_argument("--dry-run", action="store_true",
                     help="run the whole pipeline against a stub model: no GPU, no network")
    ap.add_argument("--limit", type=int, default=0,
                     help="cap on items processed per model (0 = every stage-2-built item)")
    ap.add_argument("--models", nargs="*", default=None,
                     help="substring filter(s) on the committed run's model field")
    ap.add_argument("--part", choices=sorted(PARTS), default="A",
                     help="which run-3 part's committed records to read: A=exp3a (4 options), "
                          "B=exp3b (4 options, second roster -- the AWQ and reasoning models "
                          "this experiment mainly exists for), C=exp3c (6 options)")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--n-options", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--scan", type=int, default=None)
    ap.add_argument("--data-file", type=Path, default=None)
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(args.out_dir / "probe_variants.log")
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

    total_rows = sum(len(kept) * len(CONDITIONS) for kept in reconstructed.values())
    check = budget_check(total_rows, spent=cumulative_gpu_seconds())
    log.info("budget check: %s", json.dumps(check))
    if not check["ok"]:
        log.error(
            "refusing to start: estimated %.0fs for %d (item, condition) row(s) x %d variant(s) "
            "exceeds the %.0fs remaining of the %.0fs allowance (%.0fs already spent). Lower "
            "--limit, narrow --models, or free budget first.",
            check["estimate_s"], total_rows, CALLS_PER_ROW, check["remaining_s"],
            check["allowance_s"], check["spent_s"],
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
            _rows, counts = run_probe_variants(backend, kept, args.out_dir, args.part)
        finally:
            if backend.kind != "stub":
                backend.close()

        log.info("%s: %s", backend.name,
                  json.dumps({k: v for k, v in counts.items()
                              if k not in ("completion_by_variant", "completion_by_condition",
                                           "agreement")}))
        per_model[backend.name] = counts

    summary = {
        "finished_utc": C.stamp_utc(), "part": args.part, "dry_run": False, "seed": C.SEED,
        "models": list(reconstructed), "reconstruction": recon_counts,
        "variants": {name: v.description for name, v in VARIANTS.items()},
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "per_model": per_model,
    }
    out_path = args.out_dir / "probe_variants_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
