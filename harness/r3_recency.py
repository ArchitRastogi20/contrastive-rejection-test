"""Recover the R3 edit-recency measure offline from saved stage1/stage2/stage3 JSONL.

R3 applies R1's repair to a different, non-chosen, non-gold option instead of the rival, to
catch a model that swings toward whichever profile was touched last. This module derives the
R3-edited letter (via `repair.build_conditions_with_diagnostics`'s exclusion rule, or reads it
directly from stage 3's own `edited_letter` field when present) and reports how often each
condition's choice landed on the option that condition actually edited. No GPU, no network.

    python -m harness.r3_recency --exp-dir results/exp --out results/r3_recency_summary.json

Never writes into `--exp-dir` itself.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

from . import config as C

log = logging.getLogger("r3_recency")

CONDITIONS = ("R0", "R1", "R2", "R3")

# For R0 and R3 the option that matters is the one R3 edits (untouched under R0, edited under
# R3 -- so R0 is the "nothing happened yet" baseline for exactly that option). For R1 and R2 the
# edited option is the rival itself, both conditions append their sentence to the rival's slot.
_TARGET_ROLE = {"R0": "r3_edited_option", "R1": "rival_option",
                 "R2": "rival_option", "R3": "r3_edited_option"}


class LetterUnavailable(Exception):
    """The target letter for some condition cannot be derived from the saved records.

    Raised, never worked around: a caller that catches this counts the item as unavailable and
    records the message, rather than guessing a letter it cannot derive.
    """


# --------------------------------------------------------------------------------- derivation


def derive_r3_letter(
    rival_letter: str, choice_letter: str, gold_letter: str, n_options: int
) -> str | None:
    """The letter of the option R3 edits: the first option letter, in order, that is none of
    the rival, the model's stage-1 choice, or the gold answer. Mirrors `repair.py`'s
    `excluded = {rejection.letter, choice_letter, item.gold_letter}` / `r3_idx` search exactly
    -- see this module's docstring. Returns None only if every option is excluded, which would
    mean the run itself could not have built R3 for this item either.
    """
    excluded = {rival_letter, choice_letter, gold_letter}
    for i in range(n_options):
        letter = chr(ord("A") + i)
        if letter not in excluded:
            return letter
    return None


def _recorded_targets(conds: dict[str, dict]) -> dict[str, str] | None:
    """Targets read directly off stage-3's own `edited_letter` field, when every condition that
    needs one recorded it. Returns None -- meaning the caller should fall back to
    `resolve_targets`'s derivation -- for run 1's records, which predate that field entirely and
    so never carry it; this is what keeps this module producing run 1's own numbers unchanged
    (see test_r3_recency.py::test_run1_style_records_without_edited_letter_still_derive and
    ::test_recorded_edited_letter_is_preferred_when_present).

    R0 has no edit of its own -- its target is still R3's edited letter, the "before the edit"
    reading of the same option, exactly as `resolve_targets` treats it.
    """
    r1 = conds["R1"].get("edited_letter")
    r2 = conds["R2"].get("edited_letter")
    r3 = conds["R3"].get("edited_letter")
    if not r1 or not r2 or not r3:
        return None
    return {"R0": r3, "R1": r1, "R2": r2, "R3": r3}


def resolve_targets(stage1_row: dict, rival_letter: str | None) -> dict[str, str]:
    """The target letter for each of R0-R3, or raise `LetterUnavailable` with a plain reason.

    R1 and R2's target is the rival letter directly. R0 and R3's target is the derived
    R3-edited letter -- computed once here and reused for both, since R0 is only interesting as
    the "before the edit" reading of the same option R3 edits.
    """
    if not rival_letter:
        raise LetterUnavailable("rival_letter missing or inconsistent across the R0-R3 records")

    choice_letter = stage1_row.get("choice")
    if not choice_letter:
        raise LetterUnavailable("stage-1 choice is missing or was unreadable")

    gold_letter = stage1_row.get("gold_letter")
    if not gold_letter:
        raise LetterUnavailable("gold_letter is missing from the stage-1 record")

    option_titles = stage1_row.get("option_titles")
    if not option_titles:
        raise LetterUnavailable("option_titles is missing from the stage-1 record")

    r3_letter = derive_r3_letter(rival_letter, choice_letter, gold_letter, len(option_titles))
    if r3_letter is None:
        raise LetterUnavailable("every option letter is excluded; no R3 target exists")

    return {"R0": r3_letter, "R1": rival_letter, "R2": rival_letter, "R3": r3_letter}


# ----------------------------------------------------------------------------------- loading


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _model_label(model_path: str) -> str:
    """The short model name for display -- stage rows store the full weights path."""
    return Path(model_path).name


def _load_stage(exp_dir: Path, stage: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(exp_dir.glob(f"{stage}_*.jsonl")):
        rows += load_jsonl(path)
    return rows


# --------------------------------------------------------------------------------- aggregation


def _empty_counter() -> collections.Counter:
    return collections.Counter(n=0, chose_edited_option=0, chose_other=0, unreadable=0)


def analyse(exp_dir: Path) -> dict:
    """Pooled and per-model counts of how often each condition's choice landed on the option
    that condition (or, for R0, the sibling condition R3) actually edited.

    Iterates the stage-2 rows that report `built: True` -- exactly the items a full R0-R3 set
    exists for -- and for each one, looks up its stage-1 row (for `choice`, `gold_letter`,
    `option_titles`) and its four stage-3 rows (for `rival_letter` and each condition's actual
    `choice`). Any item missing a needed piece is counted once under `items_unavailable` with a
    stated reason and excluded from every condition's counts -- never given a guessed letter.
    """
    stage1_by_key = {(r["model"], r["item_id"]): r for r in _load_stage(exp_dir, "stage1")}
    stage2_rows = _load_stage(exp_dir, "stage2")

    stage3_by_key: dict[tuple[str, str], dict[str, dict]] = collections.defaultdict(dict)
    for r in _load_stage(exp_dir, "stage3"):
        stage3_by_key[(r["model"], r["item_id"])][r["condition"]] = r

    per_model: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: {c: _empty_counter() for c in CONDITIONS}
    )
    unavailable_items: list[dict] = []
    consistency_warnings: list[str] = []
    items_considered = 0

    for row2 in stage2_rows:
        if not row2.get("built"):
            continue
        items_considered += 1
        key = (row2["model"], row2["item_id"])
        model_label = _model_label(row2["model"])

        conds = stage3_by_key.get(key, {})
        if not all(c in conds for c in CONDITIONS):
            unavailable_items.append({
                "model": model_label, "item_id": row2["item_id"],
                "reason": "not all four R0-R3 stage-3 records are present for this item",
            })
            continue

        rival_letters = {conds[c].get("rival_letter") for c in CONDITIONS}
        rival_letter = next(iter(rival_letters)) if len(rival_letters) == 1 else None

        # Prefer the letter stage 3 itself recorded (run 2+) over re-deriving it; fall back to
        # derivation for run 1's records, which never had the field at all.
        targets = _recorded_targets(conds)
        if targets is None:
            stage1_row = stage1_by_key.get(key)
            if stage1_row is None:
                unavailable_items.append({
                    "model": model_label, "item_id": row2["item_id"],
                    "reason": "no matching stage-1 record for this (model, item_id)",
                })
                continue

            try:
                targets = resolve_targets(stage1_row, rival_letter)
            except LetterUnavailable as exc:
                unavailable_items.append({
                    "model": model_label, "item_id": row2["item_id"], "reason": str(exc),
                })
                continue

        # Verify, don't trust: R1 and R2's target is the rival letter, so recomputing
        # choice == target here should always agree with stage3's own stored
        # `chosen_is_repaired_rival` flag. A disagreement would mean this module's join key or
        # the run's own field disagree about which record belongs to which item -- worth
        # surfacing rather than silently trusting a stored field that was never re-derived.
        for c in ("R1", "R2"):
            choice = conds[c].get("choice")
            stored = conds[c].get("chosen_is_repaired_rival")
            recomputed = None if choice is None else (choice == targets[c])
            if stored is not None and recomputed is not None and stored != recomputed:
                consistency_warnings.append(
                    f"{model_label}/{row2['item_id']}/{c}: stored chosen_is_repaired_rival="
                    f"{stored} disagrees with recomputed {recomputed}"
                )

        bucket = per_model[model_label]
        for c in CONDITIONS:
            choice = conds[c].get("choice")
            counter = bucket[c]
            counter["n"] += 1
            if choice is None:
                counter["unreadable"] += 1
            elif choice == targets[c]:
                counter["chose_edited_option"] += 1
            else:
                counter["chose_other"] += 1

    pooled = {c: _empty_counter() for c in CONDITIONS}
    for bucket in per_model.values():
        for c in CONDITIONS:
            pooled[c].update(bucket[c])

    def _as_dict(counters: dict[str, collections.Counter]) -> dict:
        out = {}
        for c in CONDITIONS:
            n = counters[c]["n"]
            out[c] = {
                "target_role": _TARGET_ROLE[c],
                "n": n,
                "chose_edited_option": counters[c]["chose_edited_option"],
                "chose_other": counters[c]["chose_other"],
                "unreadable": counters[c]["unreadable"],
                "rate": (counters[c]["chose_edited_option"] / n) if n else None,
            }
        return out

    return {
        "exp_dir": str(exp_dir),
        "items_considered": items_considered,
        "items_derived": items_considered - len(unavailable_items),
        "items_unavailable": unavailable_items,
        "consistency_warnings": consistency_warnings,
        "pooled": _as_dict(pooled),
        "per_model": {m: _as_dict(counters) for m, counters in sorted(per_model.items())},
    }


# --------------------------------------------------------------------------------------- CLI


def _print_table(summary: dict) -> None:
    header = f"{'':26}" + "".join(f"{c:>16}" for c in CONDITIONS)
    print(header)

    def row(label: str, block: dict) -> None:
        cells = []
        for c in CONDITIONS:
            d = block[c]
            rate = "n/a" if d["rate"] is None else f"{d['rate']:.1%}"
            cells.append(f"{d['chose_edited_option']}/{d['n']} {rate}")
        print(f"{label:26}" + "".join(f"{cell:>16}" for cell in cells))

    for model, block in summary["per_model"].items():
        row(model, block)
    row("pooled", summary["pooled"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recover the R3 recency measure from saved stage1/2/3 records"
    )
    ap.add_argument("--exp-dir", type=Path, default=C.RESULTS_DIR / "exp")
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR / "r3_recency_summary.json")
    args = ap.parse_args(argv)

    C.setup_logging()

    exp_dir = args.exp_dir.resolve()
    if not exp_dir.is_dir():
        log.error("no such directory: %s", exp_dir)
        return 2

    out = args.out.resolve()
    try:
        out.relative_to(exp_dir)
    except ValueError:
        pass
    else:
        log.error("refusing to write into %s: that directory is run 1's record", exp_dir)
        return 2

    summary = analyse(args.exp_dir)
    if summary["items_considered"] == 0:
        log.error("no built stage-2 items found under %s", args.exp_dir)
        return 2

    # The timestamp is metadata about *when this was run*, not part of the measurement itself --
    # kept out of `analyse`'s return so that function stays a pure, testable, deterministic
    # transform of the input files (see test_r3_recency.py::test_analyse_is_deterministic).
    payload = {"generated_utc": C.stamp_utc(), **summary}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8")

    _print_table(summary)
    print(f"\n{summary['items_derived']} of {summary['items_considered']} built items had a "
          f"derivable R3-edited letter ({len(summary['items_unavailable'])} unavailable).")
    if summary["consistency_warnings"]:
        for w in summary["consistency_warnings"]:
            log.warning(w)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
