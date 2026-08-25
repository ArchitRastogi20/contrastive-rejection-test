"""Re-score saved raw responses with the current extractor. No GPU, no generation.

    python -m pilot.rescore --results-dir results

Rebuilds the item set deterministically from the seed and verifies, by item id and option
order, that the rebuilt items match the ones the run actually used; stops rather than scoring
against the wrong profiles if they do not.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

from . import config as C
from . import extract
from .data import Item, build_items, load_records, load_records_from_file

log = logging.getLogger("rescore")

FUNNEL = ("items", "unreadable_choice", "with_rejection", "specific", "specific_and_true",
          "usable")


def load_raw(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("raw_*.jsonl")):
        if "attempt1" in path.name:
            continue  # the partial run that hit the memory-leak bug is kept, not counted
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    return rows


def rebuild_items(rows: list[dict], scan: int, data_file: Path | None) -> dict[str, Item]:
    """The same items the run built, keyed by id, verified against the raw records."""
    wanted = {r["item_id"] for r in rows}
    records = (load_records_from_file(data_file) if data_file
               else load_records(C.DATASET_ID, C.DATASET_SPLIT, scan))
    items = build_items(records, n_items=len(wanted), n_options=C.N_OPTIONS, seed=C.SEED)
    built = {i.item_id: i for i in items}

    missing = wanted - built.keys()
    if missing:
        raise SystemExit(
            f"rebuilt {len(built)} items but {len(missing)} of the run's items are not among "
            f"them (e.g. {sorted(missing)[:3]}). The item set is not reproducible with "
            f"seed={C.SEED} and scan={scan}: rescoring would score against the wrong profiles."
        )

    # the option order matters: a rejection is identified against the options as presented
    for r in rows:
        expected = r["option_titles"]
        actual = [o.title for o in built[r["item_id"]].options]
        if expected != actual:
            raise SystemExit(
                f"item {r['item_id']} rebuilt with a different option order:\n"
                f"  run:      {expected}\n  rebuilt:  {actual}"
            )
    return built


def rescore(rows: list[dict], items: dict[str, Item]) -> tuple[dict, list[dict]]:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    changed: list[dict] = []

    for r in rows:
        item = items[r["item_id"]]
        analysis = extract.analyse(r["response"], item)
        key = f"{r['model']}|{r['arm']}"
        c = counts[key]
        c["items"] += 1

        fresh = []
        for rej in analysis.rejections:
            fresh.append(
                {
                    "title": rej.title,
                    "attribute": rej.attribute,
                    "cue": rej.cue,
                    "complaint_is_true": extract.complaint_is_true(item, rej),
                    "usable": extract.is_usable(item, rej),
                    "sentence": rej.sentence,
                }
            )

        if analysis.choice is None:
            c["unreadable_choice"] += 1
        if fresh:
            c["with_rejection"] += 1
        if any(x["attribute"] for x in fresh):
            c["specific"] += 1
        if any(x["attribute"] and x["complaint_is_true"] for x in fresh):
            c["specific_and_true"] += 1
        if any(x["usable"] for x in fresh):
            c["usable"] += 1

        before = {(x["title"], x["attribute"], x["usable"]) for x in r["rejections"]}
        after = {(x["title"], x["attribute"], x["usable"]) for x in fresh}
        if before != after:
            changed.append({"item_id": r["item_id"], "model": r["model"], "arm": r["arm"],
                            "before": sorted(map(str, before)), "after": sorted(map(str, after))})

    return {k: dict(v) for k, v in counts.items()}, changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-score saved responses with this extractor")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--scan", type=int, default=1600)
    ap.add_argument("--data-file", type=Path, default=None)
    args = ap.parse_args(argv)

    C.setup_logging()
    rows = load_raw(args.results_dir)
    if not rows:
        log.error("no raw_*.jsonl in %s", args.results_dir)
        return 2
    log.info("re-scoring %d saved responses", len(rows))

    items = rebuild_items(rows, args.scan, args.data_file)
    counts, changed = rescore(rows, items)

    old = json.loads((args.results_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    payload = {
        "rescored_utc": C.stamp_utc(),
        "source_summary": "pilot_summary.json",
        "note": "same responses, current extractor; no generation was run",
        "records_whose_rejections_changed": len(changed),
        "gate_threshold": C.GATE_USABLE_ITEMS,
        "best_usable_count": max((v["usable"] for v in counts.values()), default=0),
        "per_run_before": {k: {f: v.get(f) for f in FUNNEL}
                           for k, v in old.get("per_run", {}).items() if isinstance(v, dict)},
        "per_run_after": counts,
        "changed_records": changed,
    }
    payload["gate_passed"] = payload["best_usable_count"] >= C.GATE_USABLE_ITEMS
    out = args.results_dir / "rescored_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'model|arm':52} {'was':>18}  {'now':>18}")
    for key in sorted(counts):
        was = payload["per_run_before"].get(key, {})
        now = counts[key]
        fmt = lambda d: (f"{d.get('specific', 0):>2}sp {d.get('specific_and_true', 0):>2}tr "
                         f"{d.get('usable', 0):>2}us")
        print(f"{key:52} {fmt(was):>18}  {fmt(now):>18}")
    print(f"\ngate: best usable {payload['best_usable_count']} of 40, threshold "
          f"{C.GATE_USABLE_ITEMS} -> {'PASS' if payload['gate_passed'] else 'FAIL'}")
    print(f"{len(changed)} of {len(rows)} records changed classification")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
