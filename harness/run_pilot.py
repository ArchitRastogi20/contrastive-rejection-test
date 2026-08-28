"""Run the contrastive-rejection pilot.

    python -m harness.run_pilot --inspect-schema        # look at the data before trusting it
    python -m harness.run_pilot --dry-run              # whole pipeline, stub model, no GPU
    python -m harness.run_pilot                        # the real thing

Per model and per arm, measures how often an explanation rejects a rival option, how often
that rejection names a concrete attribute, how often the complaint is true of the profile, and
how often the missing attribute can be sourced from a sibling profile (the gate; see the
experiment design doc). Every GPU second is committed to code/results/gpu_ledger.csv, including
on a crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path

from . import config as C
from . import extract, prompts
from .data import (
    Item,
    build_items,
    describe_schema,
    load_records,
    load_records_from_file,
)
from .models import (
    Backend,
    ModelUnavailable,
    StubBackend,
    get_backend,
    preferred_dtype,
    resolve_profile,
)
from .watchdog import Watchdog, commit_gpu_seconds, cumulative_gpu_seconds

log = logging.getLogger("harness")


# --------------------------------------------------------------------------- stub replies


_OPTION_LINE = re.compile(r"^([A-D])\)\s+(.+)$", re.MULTILINE)


class _StubReplies:
    """Canned replies that read the prompt they were given.

    An earlier version hard-coded the letter "A", which silently pointed at whichever candidate
    the deterministic shuffle happened to put first -- so the dry run exercised the plumbing but
    measured nothing. The stub now names real candidates from the rendered prompt and cycles
    three shapes: a concrete rejection, a vague one, and a second concrete one.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, index: int, chat) -> str:
        user = next(m["content"] for m in reversed(list(chat)) if m["role"] == "user")
        options = _OPTION_LINE.findall(user)
        if not options:
            return "I cannot tell from these profiles."

        # prefer a candidate whose profile mentions a birth, so the concrete shapes below
        # describe a defect that is genuinely absent from the candidate they name
        chosen = next(
            (
                (letter, title)
                for letter, title in options
                if "born" in _profile_of(user, letter).casefold()
            ),
            options[0],
        )
        other = next(((l, t) for l, t in options if l != chosen[0]), None)

        shape = self.calls % 3
        self.calls += 1
        if shape == 1 or other is None:
            return (
                f"Answer: {chosen[0]}) {chosen[1]}.\n"
                "That profile matches the question directly. The others are less relevant."
            )
        defect = "never gives a date of birth" if shape == 0 else "lacks any mention of a mother"
        return (
            f"Answer: {chosen[0]}) {chosen[1]}.\n"
            f"{chosen[1]} matches the question. "
            f"It is not {other[1]}, because that profile {defect}."
        )


def _profile_of(user: str, letter: str) -> str:
    """The profile text that follows an option line in the rendered prompt."""
    block = user.split(f"\n{letter}) ", 1)
    if len(block) < 2:
        return ""
    return block[1].split("\n", 1)[1].split("\n\n", 1)[0] if "\n" in block[1] else ""


# ------------------------------------------------------------------------- prompt sizing


def compute_max_len(items: list[Item]) -> int:
    """Size the context window from the prompts we are actually going to send.

    Fixing this constant is a trap in both directions: too small silently truncates a candidate
    profile, so the model is asked about text it never saw; too large starves the KV cache, and
    the previous study measured real throughput loss from sequence preemption when it did. So
    measure it instead of choosing it.

    The estimate is characters over 3.2, which over-counts for English, plus room for the
    generation. No tokenizer is loaded here because the model is not chosen yet.
    """
    longest = 0
    for item in items:
        for arm in prompts.ARMS:
            chars = sum(len(m["content"]) for m in prompts.render(item, arm))
            longest = max(longest, int(chars / 3.2))
    needed = longest + C.MAX_NEW_TOKENS + 256  # chat template and special tokens
    chosen = max(C.MAX_MODEL_LEN_FLOOR, min(C.MAX_MODEL_LEN_CAP, 1 << (needed - 1).bit_length()))
    log.info(
        "longest prompt about %d tokens; max_model_len=%d (floor %d, cap %d)",
        longest, chosen, C.MAX_MODEL_LEN_FLOOR, C.MAX_MODEL_LEN_CAP,
    )
    if needed > C.MAX_MODEL_LEN_CAP:
        log.warning(
            "prompts need about %d tokens, above the %d cap -- profiles may be truncated. "
            "Raise MAX_MODEL_LEN_CAP or lower N_OPTIONS rather than accepting this.",
            needed, C.MAX_MODEL_LEN_CAP,
        )
    return chosen


# ------------------------------------------------------------------------------ one pass


def run_arm(backend: Backend, items: list[Item], arm: str, out_dir: Path) -> dict:
    """Generate for every item in one arm, extract, and write one JSONL record per item."""
    label = f"{backend.name.split('/')[-1]}/{arm}"
    jsonl = out_dir / f"raw_{_slug(backend.name)}_{arm}.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    dog = Watchdog(
        label=label,
        total=len(items),
        budget_min=C.PER_MODEL_BUDGET_MIN,
        heartbeat=out_dir / "heartbeat.json",
    )
    dog.sample_vram()  # a baseline reading, before the first generation

    counts = {
        "items": 0, "unreadable_choice": 0, "with_rejection": 0, "specific": 0,
        "specific_and_true": 0, "usable": 0, "abort_reason": None,
    }

    try:
        with open(jsonl, "a", encoding="utf-8") as fh:
            for item in items:
                if dog.should_abort:
                    counts["abort_reason"] = dog.abort_reason
                    break

                text = backend.generate([prompts.render(item, arm)])[0]
                analysis = extract.analyse(text, item)

                rejections = []
                for rej in analysis.rejections:
                    truth = extract.complaint_is_true(item, rej)
                    source = extract.source_of_repair(item, rej)
                    rejections.append(
                        {
                            **asdict(rej),
                            "complaint_is_true": truth,
                            "repair_source_title": source.title if source else None,
                            "usable": extract.is_usable(item, rej),
                        }
                    )

                counts["items"] += 1
                if analysis.choice is None:
                    counts["unreadable_choice"] += 1
                if rejections:
                    counts["with_rejection"] += 1
                if any(r["attribute"] for r in rejections):
                    counts["specific"] += 1
                if any(r["attribute"] and r["complaint_is_true"] for r in rejections):
                    counts["specific_and_true"] += 1
                if any(r["usable"] for r in rejections):
                    counts["usable"] += 1

                fh.write(
                    json.dumps(
                        {
                            "model": backend.name,
                            "backend": backend.kind,
                            "arm": arm,
                            "item_id": item.item_id,
                            "question": item.question,
                            "gold_letter": item.gold_letter,
                            "gold_title": item.gold_title,
                            "option_titles": [o.title for o in item.options],
                            "type_match_score": item.type_match_score,
                            "choice": analysis.choice,
                            "choice_correct": (
                                None if analysis.choice is None
                                else analysis.choice == item.gold_letter
                            ),
                            "rejections": rejections,
                            "response": text,  # always kept: the extractor will be revised
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fh.flush()
                dog.tick()
    finally:
        # in a finally, so a crash still records the time it burned instead of recording zero
        counts["elapsed_min"] = round(dog.elapsed_s / 60.0, 2)
        counts["seconds_per_item"] = round(dog.rate_s, 2)
        counts["vram_peak_pct"] = round(dog.vram_peak * 100, 1)
        counts["abort_reason"] = counts["abort_reason"] or dog.abort_reason
        if backend.kind != "stub":
            commit_gpu_seconds(
                label=label, model=backend.name, backend=backend.kind,
                items=counts["items"], seconds=dog.elapsed_s,
                vram_peak=dog.vram_peak, abort_reason=dog.abort_reason,
            )

    return counts


def _slug(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")


def _source_name(args) -> str:
    """What the records actually came from -- the results file is the record of the run."""
    if args.data_file:
        return str(args.data_file)
    if args.dry_run:
        return "fixture:twowiki_sample.jsonl"
    return C.DATASET_ID


# ---------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Contrastive-rejection pilot")
    ap.add_argument("--inspect-schema", action="store_true",
                    help="print the first record's fields and exit; do this first")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the whole pipeline against a stub model: no GPU, no network")
    ap.add_argument("--limit", type=int, default=C.N_ITEMS)
    ap.add_argument("--scan", type=int, default=0,
                    help="records to scan for usable items (default: 40x limit)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--profile", default="auto",
                    help="model roster: auto | 3090ti | t4. auto picks by the card's VRAM")
    ap.add_argument("--arms", nargs="*", default=list(prompts.ARMS))
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--no-mirror", action="store_true",
                    help="do not substitute an ungated mirror for a gated repository")
    ap.add_argument("--data-file", type=Path, default=None,
                    help="local JSON/JSONL dump instead of the hub")
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.setup_logging(args.out_dir / "run.log")
    log.info("started %s on %s", C.stamp_utc(), platform.platform())

    scan = args.scan or args.limit * 40
    if args.data_file:
        records = load_records_from_file(args.data_file)
    elif args.dry_run:
        records = load_records_from_file(C.FIXTURE_DIR / "twowiki_sample.jsonl")
    else:
        log.info("loading %s [%s], scanning %d records", C.DATASET_ID, C.DATASET_SPLIT, scan)
        records = load_records(C.DATASET_ID, C.DATASET_SPLIT, scan)

    if args.inspect_schema:
        print(describe_schema(records))
        return 0

    items = build_items(records, n_items=args.limit, n_options=C.N_OPTIONS, seed=C.SEED)
    log.info("built %d items from %d records", len(items), len(records))
    if not items:
        log.error("no usable items. Run --inspect-schema: the context field shape has changed.")
        return 2
    if len(items) < args.limit:
        log.warning("wanted %d items, got %d -- raise --scan", args.limit, len(items))

    max_len = compute_max_len(items)
    profile = resolve_profile(args.profile)
    if profile not in C.MODEL_PROFILES:
        log.error("unknown profile %r; expected one of %s", profile, sorted(C.MODEL_PROFILES))
        return 2
    model_names = args.models or (
        ["stub"] if args.dry_run else C.MODEL_PROFILES[profile]
    )
    summary: dict[str, dict] = {}

    for name in model_names:
        if args.dry_run:
            backend: Backend = StubBackend(_StubReplies(), name="stub")
        else:
            try:
                backend = get_backend(
                    name, kind=args.backend, max_model_len=max_len,
                    allow_mirror=not args.no_mirror,
                )
            except ModelUnavailable as exc:
                log.error("skipping %s: %s", name, exc)
                log.warning("ungated substitute available: --models %s", C.SUBSTITUTE_MODEL)
                summary[name] = {"error": str(exc)}
                continue

        if backend.name != name:
            log.warning(
                "running %s in place of %s -- recorded in every result row", backend.name, name
            )
        log.info("%s ready via the %s backend", backend.name, backend.kind)

        for arm in args.arms:
            counts = run_arm(backend, items, arm, args.out_dir)
            counts["requested_model"] = name
            summary[f"{backend.name}|{arm}"] = counts
            log.info("%s / %s: %s", backend.name, arm, json.dumps(counts))
        backend.close()

    gate = max(
        (v.get("usable", 0) for v in summary.values() if isinstance(v, dict)), default=0
    )
    payload = {
        "finished_utc": C.stamp_utc(),
        "dataset": _source_name(args),
        "split": C.DATASET_SPLIT,
        "seed": C.SEED,
        "temperature": C.TEMPERATURE,
        "max_model_len": max_len,
        "gpu_mem_utilization": C.VLLM_GPU_MEM_UTILIZATION,
        "profile": profile,
        "dtype": preferred_dtype(),
        "n_items": len(items),
        "n_options": C.N_OPTIONS,
        "dry_run": args.dry_run,
        "gate_threshold": C.GATE_USABLE_ITEMS,
        "best_usable_count": gate,
        "gate_passed": gate >= C.GATE_USABLE_ITEMS,
        "cumulative_gpu_seconds": round(cumulative_gpu_seconds(), 1),
        "per_run": summary,
    }
    (args.out_dir / "pilot_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(
        "gate: best usable %d of %d items, threshold %d -> %s",
        gate, len(items), C.GATE_USABLE_ITEMS, "PASS" if payload["gate_passed"] else "FAIL",
    )
    log.info("wrote %s", args.out_dir / "pilot_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
