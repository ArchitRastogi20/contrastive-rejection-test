"""Audit whether forced letter-probe completion varies by experimental condition.

Offline audit over already-recorded stage-3 JSONL: no model, no service call, no new
generation. Reports completion by model, part, and condition, plus the number of items
retained by each paired continuous contrast, using a within-item permutation test of the
largest condition-rate spread.

    python -m pilot.audit_probe_missingness
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .config import REPO_ROOT

SEED = 20260822
RESAMPLES = 5_000
CONDITIONS = ("R0", "R1", "R2", "R3", "R4")
CONTRASTS = (("R1", "R2"), ("R3", "R4"), ("R1", "R3"), ("R2", "R4"))
PARTS = {"A": "exp3a", "B": "exp3b", "C": "exp3c"}


def load_part(results: Path, part: str) -> dict[str, dict[str, dict[str, bool]]]:
    """Return completion flags keyed as ``model -> item -> condition``."""
    out: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted((results / PARTS[part]).glob("stage3_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                probe = row.get("letter_probe") or {}
                out[row["model"]][row["item_id"]][row["condition"]] = bool(probe.get("complete"))
    return {model: dict(items) for model, items in out.items()}


def short_model(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def condition_counts(items: dict[str, dict[str, bool]]) -> dict[str, dict[str, int]]:
    """Count complete and incomplete probe reads for every condition."""
    out = {condition: {"complete": 0, "incomplete": 0} for condition in CONDITIONS}
    for per_item in items.values():
        for condition in CONDITIONS:
            if condition not in per_item:
                continue
            key = "complete" if per_item[condition] else "incomplete"
            out[condition][key] += 1
    return out


def rate_spread(counts: dict[str, dict[str, int]]) -> float | None:
    rates = []
    for condition in CONDITIONS:
        row = counts[condition]
        total = row["complete"] + row["incomplete"]
        if total:
            rates.append(row["complete"] / total)
    return max(rates) - min(rates) if rates else None


def paired_complete_counts(items: dict[str, dict[str, bool]]) -> dict[str, int]:
    """Count items with readable probes on both sides of each continuous contrast."""
    return {
        f"{left}-{right}": sum(
            per_item.get(left) is True and per_item.get(right) is True
            for per_item in items.values()
        )
        for left, right in CONTRASTS
    }


def permutation_p_value(
    items: dict[str, dict[str, bool]], *, seed: int = SEED, resamples: int = RESAMPLES,
) -> dict[str, float | int | None]:
    """Test the observed maximum condition-completion spread by within-item relabelling.

    The null preserves each item's count of complete reads, so it does not assume that
    rows are independent.  It only asks whether complete reads are allocated to labels
    more unevenly than expected if those labels had no association with completion.
    """
    complete_vectors = []
    for per_item in items.values():
        if all(condition in per_item for condition in CONDITIONS):
            complete_vectors.append([per_item[condition] for condition in CONDITIONS])
    if not complete_vectors:
        return {"observed_spread": None, "p_value": None, "n_items": 0, "resamples": resamples}

    observed = rate_spread(condition_counts(items))
    assert observed is not None
    rng = random.Random(seed)
    at_least = 0
    for _ in range(resamples):
        totals = [0] * len(CONDITIONS)
        for vector in complete_vectors:
            shuffled = vector[:]
            rng.shuffle(shuffled)
            for index, complete in enumerate(shuffled):
                totals[index] += int(complete)
        spread = (max(totals) - min(totals)) / len(complete_vectors)
        if spread >= observed - 1e-15:
            at_least += 1
    return {
        "observed_spread": observed,
        "p_value": (at_least + 1) / (resamples + 1),
        "n_items": len(complete_vectors),
        "resamples": resamples,
    }


def build_report(results: Path) -> str:
    lines = [
        "# Condition-specific letter-probe missingness audit",
        "",
        "## Scope",
        "",
        "This CPU-only audit reads the committed run-3 stage-3 JSONL (about 37 MB). It uses no API,",
        "no model, no network, and no new generation. It tests only whether the forced single-token",
        "letter probe's completion varies by condition; it does not test a scientific effect.",
        "",
        f"Seed: `{SEED}`. Within-item permutation resamples: `{RESAMPLES:,}`. The statistic is the",
        "maximum difference in completion rate across R0–R4. A non-significant result is not evidence",
        "that missingness is random; it is only insufficient evidence of condition-specific completion.",
        "",
        "## Completion by model, part, and condition",
        "",
        "| part | model | R0 | R1 | R2 | R3 | R4 | largest rate spread | permutation p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    findings = []
    for part in PARTS:
        for model, items in sorted(load_part(results, part).items()):
            counts = condition_counts(items)
            test = permutation_p_value(items)
            cells = []
            for condition in CONDITIONS:
                row = counts[condition]
                total = row["complete"] + row["incomplete"]
                cells.append(f"{row['complete']}/{total}")
            spread = test["observed_spread"]
            p_value = test["p_value"]
            spread_text = "n/a" if spread is None else f"{spread:.3f}"
            p_text = "n/a" if p_value is None else f"{p_value:.4f}"
            lines.append(
                f"| {part} | {short_model(model)} | " + " | ".join(cells)
                + f" | {spread_text} | {p_text} |"
            )
            if spread is not None and 0 < spread < 1:
                findings.append((part, short_model(model), test, paired_complete_counts(items)))

    lines += [
        "",
        "## Paired continuous-contrast availability for partially complete probes",
        "",
        "| part | model | R1−R2 | R3−R4 | R1−R3 | R2−R4 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for part, model, _test, pairs in findings:
        lines.append(
            f"| {part} | {model} | {pairs['R1-R2']} | {pairs['R3-R4']} | "
            f"{pairs['R1-R3']} | {pairs['R2-R4']} |"
        )

    lines += ["", "## Interpretation", ""]
    if not findings:
        lines.append("All probes are either fully complete or fully incomplete, so no partial-completion audit applies.")
    else:
        for part, model, test, pairs in findings:
            p_value = test["p_value"]
            assert isinstance(p_value, float)
            direction = "evidence of condition-specific completion" if p_value < 0.05 else "no detected condition-specific completion"
            lines.append(
                f"- **Part {part}, {model}:** {direction} under the within-item permutation audit "
                f"(spread {test['observed_spread']:.3f}, p={p_value:.4f}, n={test['n_items']}). "
                f"The four paired continuous contrasts retain {pairs['R1-R2']}, {pairs['R3-R4']}, "
                f"{pairs['R1-R3']}, and {pairs['R2-R4']} items respectively."
            )
    lines += [
        "",
        "The audit should be reported as a coverage diagnostic. It cannot recover the missing probability",
        "reads, establish missing-at-random assumptions, or make probability results comparable to the greedy",
        "choice measure. The appropriate manuscript language remains: continuous results are a complete-case,",
        "prompt-specific operationalization with model- and condition-specific coverage reported alongside it.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    global RESAMPLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    RESAMPLES = args.resamples
    out = args.out or (REPO_ROOT / "code" / "probe_missingness_audit_report.txt")
    out.write_text(build_report(args.results), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
