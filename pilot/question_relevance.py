"""How many built items ask a question about the very attribute the repair inserts?

The repair sentence need not be true of the target entity: the claim under test is only that the
profile omits the named attribute, and any sentence stating it falsifies that claim. That is
sound for the text-level claim, and it has a consequence no gate checks. When the question is a
date or order comparison ("who was born first", "which film came out first") and the repair
inserts a date, a false inserted value can change which option is actually correct. The observed
movement would then follow from the question's own answer changing rather than from anything
about the model's stated reason.

This counts that overlap. It is deliberately a plain lexical rule over the question text, printed
below so the classification can be audited rather than trusted, and it is committed so the
figure in the paper regenerates:

    python -m pilot.question_relevance

No GPU, no network, no model: it reads committed stage-1 and stage-2 records only.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from . import config as C

# Comparative questions in this corpus are phrased a small number of ways. Each alternative is a
# whole phrase rather than a bare word, because "later" and "earlier" alone also appear in
# non-comparative questions ("who died later than the war") often enough to matter.
ORDER_QUESTION = re.compile(
    r"came out first|born first|died first|which film was released"
    r"|who was born|who died|earlier|later|more recently",
    re.IGNORECASE,
)

# The repair attributes whose value a date comparison turns on.
DATE_ATTRIBUTES = frozenset({"date_of_birth", "date_of_death"})

PARTS = ("exp3a", "exp3b", "exp3c")


def question_by_item(results: Path) -> dict[tuple[str, str], str]:
    """`(item_id, model) -> question`, from whichever stage-1 records the parts were built on.

    Part A replays `exp2`'s stage 1 rather than carrying its own, which its
    `experiment_summary.json` records; reading both directories covers every part without
    special-casing.
    """
    out: dict[tuple[str, str], str] = {}
    for part in PARTS + ("exp2",):
        for path in glob.glob(str(results / part / "stage1*.jsonl")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    out.setdefault((row["item_id"], row["model"]), row.get("question", ""))
    return out


def count(results: Path) -> dict:
    """Built items, how many pose an order comparison, and how many of those get a date repair."""
    questions = question_by_item(results)
    built = order = order_and_date = 0
    unmatched = 0
    by_attribute: Counter[str] = Counter()
    for part in PARTS:
        for path in glob.glob(str(results / part / "stage2*.jsonl")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not row.get("built"):
                        continue
                    built += 1
                    question = questions.get((row["item_id"], row["model"]))
                    if question is None:
                        unmatched += 1
                        continue
                    if not ORDER_QUESTION.search(question):
                        continue
                    order += 1
                    if row.get("attribute") in DATE_ATTRIBUTES:
                        order_and_date += 1
                        by_attribute[row.get("attribute")] += 1
    return {
        "built": built,
        "order_question": order,
        "order_question_share": order / built if built else 0.0,
        "order_question_with_date_repair": order_and_date,
        "by_attribute": dict(by_attribute),
        "built_without_a_matching_stage1_row": unmatched,
    }


def sample(results: Path, n: int = 10) -> list[tuple[bool, str]]:
    """A few real questions with their classification, so the rule can be eyeballed."""
    out: list[tuple[bool, str]] = []
    for (_, _), question in question_by_item(results).items():
        if not question:
            continue
        out.append((bool(ORDER_QUESTION.search(question)), question))
        if len(out) >= n:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=C.RESULTS_DIR, type=Path)
    args = ap.parse_args(argv)

    got = count(args.results)
    print(f"built items                              {got['built']}")
    print(f"  question is a date/order comparison    {got['order_question']}"
          f"  ({got['order_question_share']:.1%})")
    print(f"  of those, repair inserts a date        {got['order_question_with_date_repair']}")
    print(f"  by attribute                           {got['by_attribute']}")
    if got["built_without_a_matching_stage1_row"]:
        print(f"  built rows with no stage-1 match       "
              f"{got['built_without_a_matching_stage1_row']}")
    print("\nclassification rule:")
    print(f"  {ORDER_QUESTION.pattern}")
    print("\nsample, so the rule can be audited rather than trusted:")
    for is_order, question in sample(args.results):
        print(f"  [{'order' if is_order else '  -  '}] {question[:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
