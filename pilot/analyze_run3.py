"""Offline analyses of run 3, computed from the committed stage-3 JSONL alone.

No GPU, no network, no model, no new generation: every number is a deterministic function of
files already in ``code/results/``.

    python -m pilot.analyze_run3                    # writes the run-3 analysis report
    python -m pilot.analyze_run3 --self-check       # arithmetic checks, no data needed

Computes the paired-data audit table, the rank-margin decomposition, the roster interaction,
Holm-adjusted multiplicity, and two figures the paper states on the wrong population.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

from . import extract
from .config import REPO_ROOT
from .data import Entity
from .run_experiment import bootstrap_ci_mean_diff, mcnemar

# The seed every bootstrap in this file uses, recorded here once so a reader never has to hunt
# for it. Same discipline as the run itself: a resample that cannot be reproduced is not evidence.
SEED = 20260822
RESAMPLES = 10000

PARTS = {
    "A": ("exp3a", "4 options, original roster"),
    "B": ("exp3b", "4 options, second roster"),
    "C": ("exp3c", "6 options, original roster"),
}
CONTRASTS = [
    ("R1", "R2", "content at the rival"),
    ("R3", "R4", "content at the third option"),
    ("R1", "R3", "location, relevant content"),
    ("R2", "R4", "location, irrelevant content"),
]


# --------------------------------------------------------------------------------- loading


def _rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_option_titles(results: Path, part: str) -> dict[tuple[str, str], list[str]]:
    """``(model, item_id) -> option_titles`` from that part's own stage-1 records.

    Stage-3 rows carry no option titles, only the fixed ``response`` text choice parsing needs
    them for. Part B and Part C ran their own stage 1, so their titles live under
    ``exp3b``/``exp3c``; Part A did not -- ``exp3a/experiment_summary.json`` records ``"dataset":
    "stage1 replay: results/exp2/..."``, so Part A's titles are read from ``exp2`` instead,
    which is where its stage-1 files actually are.
    """
    directory, _ = PARTS[part]
    stage1_dir = results / ("exp2" if part == "A" else directory)
    titles: dict[tuple[str, str], list[str]] = {}
    for path in sorted(stage1_dir.glob("stage1_*.jsonl")):
        for row in _rows(path):
            titles[(row["model"], row["item_id"])] = row["option_titles"]
    return titles


def _recompute_choice_fields(row: dict, option_titles: list[str]) -> None:
    """Overwrite ``choice``/``chosen_is_edited`` in place, re-derived from the row's own raw
    ``response`` with the fixed-letter-range parser.

    ``extract.parse_choice``'s two letter regexes used to be hard-capped at ``[A-D]``, so a
    six-option Part C item could never have a chosen letter of E or F read out of it. That is
    fixed in ``extract.py`` itself; this re-derives the stage-3 file's ``choice`` (written at
    run time with the buggy parser) the same way, rather than trusting what is on disk.
    ``parse_choice`` only reads ``item.options[i].title`` and the option count, so a bare
    options-only stand-in is enough -- stage-3 rows carry no question/answer/gold_title to build
    a real ``Item`` from, and ``parse_choice`` never looks at those fields anyway.

    ``edited_letter`` is left untouched: it comes from the edit itself
    (``run_experiment.run_stage3``'s ``edited_letter_by_condition``), not from parsing free
    text, so the bug never touched it. ``chosen_is_edited`` is recomputed with the exact same
    rule ``run_stage3`` used -- ``None`` if either side is unreadable/absent, else the equality.
    """
    item_stub = SimpleNamespace(options=[Entity(title=t, sentences=[]) for t in option_titles])
    choice = extract.parse_choice(row.get("response", ""), item_stub)
    edited_letter = row.get("edited_letter")
    row["choice"] = choice
    row["chosen_is_edited"] = (
        None if choice is None or edited_letter is None else choice == edited_letter
    )


def load_part(results: Path, part: str) -> dict[str, dict[str, dict[str, dict]]]:
    """``{model: {item_id: {condition: row}}}`` for one part's stage-3 output.

    Keyed by model first because every contrast below is reported per model before it is
    pooled: run 2's audit found one model carrying most of a pooled effect, and a pooled-only
    table is exactly what hid that.

    Each row's ``choice``/``chosen_is_edited`` are overwritten here, re-derived from the raw
    ``response`` (see ``_recompute_choice_fields``), before anything downstream ever sees the
    row -- so every reader of this dict's output, including ``discrete_outcomes``, gets the
    corrected value without having to know it was corrected. A stage-3 row whose
    ``(model, item_id)`` has no stage-1 titles is a coverage hole, not a row to skip silently,
    so it raises rather than dropping the row.
    """
    directory, _ = PARTS[part]
    titles = _load_option_titles(results, part)
    out: dict[str, dict[str, dict[str, dict]]] = {}
    for path in sorted((results / directory).glob("stage3_*.jsonl")):
        for row in _rows(path):
            key = (row["model"], row["item_id"])
            if key not in titles:
                raise KeyError(
                    f"part {part}: no stage-1 option_titles for (model={key[0]!r}, "
                    f"item_id={key[1]!r}) -- checked {'exp2' if part == 'A' else directory}"
                )
            _recompute_choice_fields(row, titles[key])
            out.setdefault(row["model"], {}).setdefault(row["item_id"], {})[row["condition"]] = row
    return out


def short_model(name: str) -> str:
    return name.rstrip("/").split("/")[-1]


# ------------------------------------------------------------------ the two paired measures


def discrete_outcomes(items: dict[str, dict[str, dict]]) -> dict[str, dict[str, bool | None]]:
    """``chosen_is_edited`` per item per condition, the field ``mcnemar`` already consumes.

    By the time a row reaches here, ``load_part`` has already overwritten ``chosen_is_edited``
    with the value re-derived from the raw response (see ``_recompute_choice_fields``) -- this
    function itself just reads whatever key is on the row, same as before.
    """
    return {
        item_id: {c: row.get("chosen_is_edited") for c, row in conds.items()}
        for item_id, conds in items.items()
    }


def _margin(probs: dict[str, float], letter: str) -> float | None:
    """The target letter's probability minus the best competitor's.

    Positive means the probe would pick this letter; the sign is what an argmax reads, and the
    magnitude is how far the edit would have to move mass to change that reading. This is the
    quantity a flip is a threshold crossing of, which raw ``p_edited`` is not.
    """
    if letter not in probs:
        return None
    others = [p for k, p in probs.items() if k != letter]
    if not others:
        return None
    return probs[letter] - max(others)


def margin_outcomes(items: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Per item per condition, the change in the target letter's margin against R0.

    Built to be read by ``continuous_contrast``-style code: each condition carries
    ``delta_margin`` alongside a ``complete`` flag, so an incomplete probe read is excluded and
    never imputed, the same rule the run's own continuous measure follows.
    """
    out: dict[str, dict[str, dict]] = {}
    for item_id, conds in items.items():
        r0 = conds.get("R0")
        if r0 is None:
            continue
        r0_probe = r0.get("letter_probe") or {}
        if not r0_probe.get("complete"):
            continue
        r0_probs = r0_probe.get("probs") or {}
        per_item: dict[str, dict] = {}
        for cond, row in conds.items():
            if cond == "R0":
                continue
            letter = row.get("edited_letter")
            probe = row.get("letter_probe") or {}
            if letter is None or not probe.get("complete"):
                continue
            here = _margin(probe.get("probs") or {}, letter)
            base = _margin(r0_probs, letter)
            if here is None or base is None:
                continue
            per_item[cond] = {
                "delta_margin": here - base,
                "margin_after": here,
                "margin_before": base,
                "chosen_is_edited": row.get("chosen_is_edited"),
                "complete": True,
            }
        if per_item:
            out[item_id] = per_item
    return out


def paired_diffs(outcomes: dict[str, dict[str, dict]], a: str, b: str, field: str) -> list[float]:
    """Per-item ``field`` at condition ``a`` minus ``field`` at condition ``b``, over items
    complete on both sides. Exposed on its own (not just folded into ``paired_mean``) so other
    statistics -- median, trimmed mean, sign counts -- can be read off exactly the same
    underlying diffs the mean is computed from, rather than a differently-filtered population."""
    diffs = []
    for per_item in outcomes.values():
        ra, rb = per_item.get(a), per_item.get(b)
        if ra is None or rb is None:
            continue
        va, vb = ra.get(field), rb.get(field)
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
    return diffs


def discordant_concordant_diffs(
    disc_outcomes: dict[str, dict[str, bool | None]],
    other_outcomes: dict[str, dict[str, dict]],
    a: str, b: str, field: str,
) -> tuple[list[float], list[float]]:
    """Split the paired ``field`` diffs between conditions ``a``/``b`` into the ``discordant``
    subset -- items whose discrete outcome (``chosen_is_edited``) differs between ``a`` and
    ``b``, exactly the items McNemar's ``b`` and ``c`` are counted from -- and the
    ``concordant`` subset -- every other item with a readable discrete outcome on both sides.

    An item is excluded from both subsets (not silently folded into one) if its discrete
    outcome is unreadable on either side, or if ``field`` itself is missing on either side --
    the same exclusion rule ``paired_diffs`` uses, applied on top of the discordance check
    rather than instead of it. This exists because McNemar and a sign count over ``field`` need
    not be reading the same population: McNemar only sees the discordant items, a plain sign
    count sees every item with a reading, and the two can disagree in direction not because
    either is wrong but because they are weighting different subpopulations.
    """
    discordant: list[float] = []
    concordant: list[float] = []
    for item_id, per_item in other_outcomes.items():
        ra, rb = per_item.get(a), per_item.get(b)
        if ra is None or rb is None:
            continue
        va, vb = ra.get(field), rb.get(field)
        if va is None or vb is None:
            continue
        disc_item = disc_outcomes.get(item_id)
        if disc_item is None:
            continue
        da, db = disc_item.get(a), disc_item.get(b)
        if da is None or db is None:
            continue
        diff = va - vb
        (discordant if bool(da) != bool(db) else concordant).append(diff)
    return discordant, concordant


def paired_mean(outcomes: dict[str, dict[str, dict]], a: str, b: str, field: str,
                *, seed: int = SEED) -> dict:
    """Paired mean of ``field`` between two conditions, over items complete on both sides."""
    diffs = paired_diffs(outcomes, a, b, field)
    result = bootstrap_ci_mean_diff(diffs, seed=seed, n_resamples=RESAMPLES)
    result["contrast"] = f"{a}-{b}"
    return result


# --------------------------------------------------------- sign / median / trimmed-mean toolkit


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _trimmed_mean(values: list[float], frac: float) -> float | None:
    """Symmetric trimmed mean: sort, drop ``frac`` of the items from each tail, average what's
    left. Falls back to the plain mean if the trim would remove everything (a tiny sample)."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    k = int(n * frac)
    kept = s[k:n - k] if n - 2 * k > 0 else s
    return sum(kept) / len(kept) if kept else None


def _sign_counts(values: list[float]) -> tuple[int, int, int]:
    """(positive, negative, exactly-zero) counts -- the sign distribution McNemar is itself a
    function of, read here off a continuous quantity instead of an argmax flip."""
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    zero = sum(1 for v in values if v == 0)
    return pos, neg, zero


def bootstrap_stat(values: list[float], stat, *, seed: int = SEED,
                    n_resamples: int = RESAMPLES, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for an arbitrary statistic of ``values``.

    Same resampling discipline as ``run_experiment.bootstrap_ci_mean_diff`` -- one full resample
    of length ``n`` at a time, drawn in order from ``random.Random(seed)``, same percentile index
    formula -- generalised beyond the mean because the point of this section is to check whether
    the mean, specifically, is what disagrees with the sign/median picture.
    """
    n = len(values)
    if n == 0:
        return {"stat": None, "ci_low": None, "ci_high": None, "n": 0, "seed": seed}
    point = stat(values)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        draws.append(stat([values[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    lo_idx = min(n_resamples - 1, max(0, int((alpha / 2) * n_resamples)))
    hi_idx = min(n_resamples - 1, max(0, int((1 - alpha / 2) * n_resamples) - 1))
    return {"stat": point, "ci_low": draws[lo_idx], "ci_high": draws[hi_idx], "n": n, "seed": seed}


# ------------------------------------------------------------------------- effect sizes


def odds_ratio(b: int, c: int) -> dict:
    """Matched-pairs OR = b/c, Wald CI on the log scale, Haldane-Anscombe only when a cell is 0.

    Matches the convention the committed interaction test used, so the released numbers and
    this script's numbers are comparable rather than merely similar.
    """
    if b == 0 or c == 0:
        bb, cc = b + 0.5, c + 0.5
        corrected = True
    else:
        bb, cc = float(b), float(c)
        corrected = False
    or_hat = bb / cc
    se = math.sqrt(1.0 / bb + 1.0 / cc)
    return {
        "or": or_hat,
        "lo": or_hat * math.exp(-1.96 * se),
        "hi": or_hat * math.exp(1.96 * se),
        "haldane_anscombe": corrected,
    }


def risk_difference(outcomes: dict[str, dict[str, bool | None]], a: str, b: str,
                    *, seed: int = SEED) -> dict:
    """Paired risk difference, bootstrap CI, over items readable on both sides."""
    diffs = []
    for per_item in outcomes.values():
        va, vb = per_item.get(a), per_item.get(b)
        if va is None or vb is None:
            continue
        diffs.append(float(bool(va)) - float(bool(vb)))
    return bootstrap_ci_mean_diff(diffs, seed=seed, n_resamples=RESAMPLES)


def discrete_contrast(outcomes: dict[str, dict[str, bool | None]], a: str, b: str) -> dict:
    mc = mcnemar(outcomes, a, b)
    rd = risk_difference(outcomes, a, b)
    orr = odds_ratio(mc["b_a_only"], mc["c_b_only"])
    return {**mc, "or": orr, "rd": rd, "n_paired": rd["n"], "contrast": f"{a}-{b}"}


# ---------------------------------------------------------------------------- multiplicity


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values. Monotone by construction, capped at 1."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


# ---------------------------------------------------------------------------------- power


def mcnemar_power(n_items: int, disc_rate: float, odds: float, alpha: float = 0.05) -> float:
    """Exact power of a two-sided exact McNemar test, by enumeration rather than simulation.

    The discordant count is Binomial(n_items, disc_rate); given it, the split between the two
    discordant cells is Binomial(D, odds/(1+odds)). Both are summed exactly, so the answer is a
    number rather than a draw, which matters because a simulated power estimate would be one
    more thing in this project that cannot be reproduced without its seed.
    """
    if not 0 < disc_rate <= 1 or odds <= 0:
        return float("nan")
    pi = odds / (1.0 + odds)
    power = 0.0
    for d in range(0, n_items + 1):
        log_pd = (math.lgamma(n_items + 1) - math.lgamma(d + 1) - math.lgamma(n_items - d + 1)
                  + d * math.log(disc_rate) + (n_items - d) * math.log1p(-disc_rate)
                  if 0 < disc_rate < 1 else (0.0 if d == n_items else float("-inf")))
        if log_pd == float("-inf"):
            continue
        p_d = math.exp(log_pd)
        if p_d < 1e-12:
            continue
        reject = 0.0
        for b in range(0, d + 1):
            k = min(b, d - b)
            tail = sum(math.comb(d, i) for i in range(0, k + 1)) / (2 ** d)
            if min(1.0, 2 * tail) <= alpha:
                reject += math.comb(d, b) * (pi ** b) * ((1 - pi) ** (d - b))
        power += p_d * reject
    return power


def independent_difference(diffs_a: list[float], diffs_b: list[float], *, seed: int = SEED) -> dict:
    """Difference between two independent samples' paired means, bootstrapped.

    Each part is resampled within itself, because Part A and Part B are different items on
    different models: pooling them into one resample would treat a between-roster difference as
    if it were within-item variation.
    """
    if not diffs_a or not diffs_b:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_a": len(diffs_a),
                "n_b": len(diffs_b)}
    mean = sum(diffs_a) / len(diffs_a) - sum(diffs_b) / len(diffs_b)
    rng = random.Random(seed)
    draws = []
    for _ in range(RESAMPLES):
        sa = sum(diffs_a[rng.randrange(len(diffs_a))] for _ in range(len(diffs_a))) / len(diffs_a)
        sb = sum(diffs_b[rng.randrange(len(diffs_b))] for _ in range(len(diffs_b))) / len(diffs_b)
        draws.append(sa - sb)
    draws.sort()
    return {"mean": mean, "ci_low": draws[int(0.025 * RESAMPLES)],
            "ci_high": draws[int(0.975 * RESAMPLES) - 1],
            "n_a": len(diffs_a), "n_b": len(diffs_b), "seed": seed}


def discrete_diffs(outcomes: dict[str, dict[str, bool | None]], a: str, b: str) -> list[float]:
    diffs = []
    for per_item in outcomes.values():
        va, vb = per_item.get(a), per_item.get(b)
        if va is None or vb is None:
            continue
        diffs.append(float(bool(va)) - float(bool(vb)))
    return diffs


# ------------------------------------------------------------------ the two stated-wrong figures


def _recompute_choice_correct(row: dict) -> bool | None:
    """Stage 1's ``choice_correct``, re-derived from the row's own raw ``response`` with the
    fixed-precedence parser, rather than trusted as written at generation time.

    The stage-1 rows carry the identical defect the stage-3 fix (``_recompute_choice_fields``)
    addresses: a response that opens "A) The Mask of Fu Manchu" and later writes "I ruled out
    candidate C) The Mysterious Dr. Fu Manchu" was read, at generation time, as choosing C --
    the confirmed bug, present at stage 1 as well as stage 3. ``gate_eight_skew`` reports
    stage-1 correctness, so it must not read that stale field.
    """
    item_stub = SimpleNamespace(
        options=[Entity(title=t, sentences=[]) for t in row["option_titles"]]
    )
    choice = extract.parse_choice(row.get("response", ""), item_stub)
    return None if choice is None else choice == row["gold_letter"]


def gate_eight_skew(results: Path, part: str) -> dict:
    """Stage-one correctness among built items against the attempted pool, per part.

    The paper states this as 62.7% against 47.4%, citing run 2's 295-item built set, and
    describes it as holding in every part. Part A replays run 2's stage 1, so its stage-1 rows
    live in ``exp2``; Parts B and C carry their own.

    The join rule, stated explicitly because it is where an earlier version of this function
    disagreed with an independent audit: a stage-2 row enters the attempted pool only if (a) a
    stage-1 row exists for the same ``(model, item_id)``, and (b) that stage-1 row's
    ``choice_correct``, re-derived by ``_recompute_choice_correct`` rather than trusted as
    written, is not ``None``. ``choice_correct`` is ``None`` exactly when stage 1's free-text
    ``choice`` could not be parsed at all -- there is no correctness signal for that item, not a
    negative one. An earlier version of this function read it with
    ``bool(row.get("choice_correct"))``, which silently coerces that ``None`` to ``False`` and
    counts an unreadable response as an *incorrect* one, inflating the attempted-pool
    denominator (and its incorrect count) with rows that were never judged. Both exclusion
    reasons -- no stage-1 row at all, and a stage-1 row with no parseable choice -- are counted
    and reported separately.
    """
    directory, _ = PARTS[part]
    stage1_dir = results / ("exp2" if part == "A" else directory)
    correct: dict[tuple[str, str], bool] = {}
    unparseable_at_stage1: set[tuple[str, str]] = set()
    for path in sorted(stage1_dir.glob("stage1_*.jsonl")):
        for row in _rows(path):
            key = (row["model"], row["item_id"])
            cc = _recompute_choice_correct(row)
            if cc is None:
                unparseable_at_stage1.add(key)
            else:
                correct[key] = bool(cc)

    built_n = built_ok = att_n = att_ok = 0
    unjoined_missing_key = unjoined_unparseable_choice = 0
    for path in sorted((results / directory).glob("stage2_*.jsonl")):
        for row in _rows(path):
            key = (row["model"], row["item_id"])
            if key in unparseable_at_stage1:
                unjoined_unparseable_choice += 1
                continue
            hit = correct.get(key)
            if hit is None:
                unjoined_missing_key += 1
                continue
            att_n += 1
            att_ok += hit
            if row.get("built"):
                built_n += 1
                built_ok += hit
    return {"built_correct": built_ok, "built_n": built_n, "attempted_correct": att_ok,
            "attempted_n": att_n,
            "unjoined_missing_key": unjoined_missing_key,
            "unjoined_unparseable_choice": unjoined_unparseable_choice,
            "unjoined": unjoined_missing_key + unjoined_unparseable_choice,
            "built_rate": built_ok / built_n if built_n else None,
            "attempted_rate": att_ok / att_n if att_n else None}


def _r0_probe_probs(row: dict | None) -> dict:
    if row is None:
        return {}
    return (row.get("letter_probe") or {}).get("probs") or {}


def _summarise_baseline(rival: list[float], third: list[float]) -> dict:
    out = {}
    for name, values in (("rival", rival), ("third", third)):
        out[name] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
    if out["rival"]["mean"] and out["third"]["mean"]:
        out["ratio"] = out["rival"]["mean"] / out["third"]["mean"]
    else:
        out["ratio"] = None
    return out


def baseline_by_target_paper(items: dict[str, dict[str, dict]]) -> dict:
    """R0 probability on each location's own target letter -- the population Table 3 and the
    prior run-3 audit's PART 1 actually used.

    Read off R1 and R3 only: R2 and R4 share their targets, so including them would duplicate
    every item rather than add one. R0's own ``letter_probe.probs`` is read directly for the
    R1/R3 row's ``edited_letter``, gated only on that letter being a key in R0's ``probs`` --
    neither R0's own ``complete`` flag nor the R1/R3 row's own probe completeness gates this. A
    probe can be ``complete: False`` (missing mass on some *other* candidate letter) while still
    reporting a real number for the one letter this asks about, and the paper's number is the
    one that keeps those rows.
    """
    rival, third = [], []
    for conds in items.values():
        r0_probs = _r0_probe_probs(conds.get("R0"))
        for cond, bucket in (("R1", rival), ("R3", third)):
            row = conds.get(cond)
            if row is None:
                continue
            letter = row.get("edited_letter")
            if letter is not None and letter in r0_probs:
                bucket.append(r0_probs[letter])
    return _summarise_baseline(rival, third)


def baseline_by_target_strict(items: dict[str, dict[str, dict]]) -> dict:
    """The same quantity as ``baseline_by_target_paper``, read instead off the run's own
    ``r0_p_target`` field, which additionally requires the R1/R3 row's *own* letter probe to be
    complete (see ``run_experiment.run_stage3``'s ``r0_p_target`` computation), not just R0's.

    A smaller, stricter population -- and the one internally consistent with the rows
    ``delta_p_edited`` and ``delta_margin`` use elsewhere in this report, since those quantities
    are themselves gated on the row's own probe being complete. It coincides with the paper
    population in Part B, where no model's probe is ever partially complete, and diverges from
    it in Parts A and C, where Mistral's probe completeness is well below 100%.
    """
    rival, third = [], []
    for conds in items.values():
        for cond, bucket in (("R1", rival), ("R3", third)):
            row = conds.get(cond)
            if row is not None and row.get("r0_p_target") is not None:
                bucket.append(row["r0_p_target"])
    return _summarise_baseline(rival, third)


# --------------------------------------------------------------------------------- reporting


def _fmt(x, places=4):
    return "n/a" if x is None else f"{x:+.{places}f}"


def _ci(d, places=4):
    if d.get("ci_low") is None:
        return "n/a"
    return f"{d['mean']:+.{places}f} [{d['ci_low']:+.{places}f}, {d['ci_high']:+.{places}f}]"


def _ci_stat(d, places=4):
    """Like ``_ci``, for a ``bootstrap_stat`` result -- point estimate under key ``stat``
    rather than ``mean``, since the statistic need not be a mean."""
    if d.get("ci_low") is None:
        return "n/a"
    return f"{d['stat']:+.{places}f} [{d['ci_low']:+.{places}f}, {d['ci_high']:+.{places}f}]"


def build_report(results: Path) -> str:
    lines: list[str] = []
    w = lines.append

    w("# Offline analyses of run 3")
    w("")
    w("Generated by `python -m pilot.analyze_run3` from the committed stage-3 JSONL in")
    w("`code/results/exp3a`, `exp3b`, `exp3c`. No GPU, no network, no generation. Every")
    w(f"bootstrap here uses seed `{SEED}` with `{RESAMPLES}` resamples, and every exact test is")
    w("the same `math.comb`-based binomial `run_experiment.mcnemar` uses.")
    w("")

    loaded = {part: load_part(results, part) for part in PARTS}
    discrete_by_part: dict[str, dict[str, bool | None]] = {}
    margins_by_part = {}

    # ------------------------------------------------------------------ 1. audit table
    w("## 1. The paired-data audit table")
    w("")
    w("Every contrast, per model and pooled, with the paired `n` each measure actually has.")
    w("The two measures do not share a denominator: the discrete read needs only a parseable")
    w("answer, the continuous read needs a complete letter probe, and a model whose probe")
    w("degrades drops out of the second while staying in the first.")
    w("")
    for part, (_, label) in PARTS.items():
        models = loaded[part]
        pooled_items = {f"{m}::{i}": conds for m, its in models.items() for i, conds in its.items()}
        w(f"### Part {part} ({label})")
        w("")
        w("| scope | contrast | n disc | b | c | exact p | OR [95% CI] | RD [95% CI] | n cont | Δp [95% CI] | n margin | Δmargin [95% CI] |")
        w("|---|---|---:|---:|---:|---:|---|---|---:|---|---:|---|")
        scopes = [(short_model(m), its) for m, its in sorted(models.items())]
        scopes.append(("**pooled**", pooled_items))
        for scope, its in scopes:
            disc = discrete_outcomes(its)
            marg = margin_outcomes(its)
            cont = {
                item_id: {c: {"delta_p_edited": row.get("delta_p_edited")}
                          for c, row in conds.items()}
                for item_id, conds in its.items()
            }
            if scope == "**pooled**":
                discrete_by_part[part] = disc
                margins_by_part[part] = marg
            for a, b, _why in CONTRASTS:
                d = discrete_contrast(disc, a, b)
                o, rd = d["or"], d["rd"]
                cm = paired_mean(cont, a, b, "delta_p_edited")
                mm = paired_mean(marg, a, b, "delta_margin")
                w(f"| {scope} | {a}−{b} | {d['n_paired']} | {d['b_a_only']} | {d['c_b_only']} "
                  f"| {d['p_exact_two_sided']:.4f} "
                  f"| {o['or']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}]{'*' if o['haldane_anscombe'] else ''} "
                  f"| {_ci(rd, 3)} | {cm['n']} | {_ci(cm)} | {mm['n']} | {_ci(mm)} |")
        w("")
    w("`*` marks an odds ratio computed with a Haldane-Anscombe correction because a discordant")
    w("cell was empty.")
    w("")

    # ------------------------------------------------------- 2. rank-margin decomposition
    w("## 2. The rank-margin decomposition")
    w("")
    w("The discrete measure is a threshold crossing of the target's margin against its best")
    w("competitor. The continuous measure is the target's own probability. Section 1's last two")
    w("columns put them side by side; this section asks the question they were added for.")
    w("")
    w("First, whether the two instruments agree at the level of a single row. If the probe's own")
    w("sign disagrees with the generated answer, the disagreement is between instruments; if it")
    w("agrees, the disagreement is only between two functionals of one distribution.")
    w("")
    w("| part | rows with both reads | probe sign agrees with generated choice | rate |")
    w("|---|---:|---:|---:|")
    for part in PARTS:
        agree = total = 0
        for per_item in margins_by_part[part].values():
            for rec in per_item.values():
                if rec["chosen_is_edited"] is None:
                    continue
                total += 1
                agree += (rec["margin_after"] > 0) == bool(rec["chosen_is_edited"])
        rate = f"{agree / total:.4f}" if total else "n/a"
        w(f"| {part} | {total} | {agree} | {rate} |")
    w("")
    w("The same rate, broken down by model and by condition: a uniform ~15% disagreement across")
    w("every model and every condition is a different finding from one model or one condition")
    w("driving all of it.")
    w("")
    w("| part | model | condition | rows | probe sign agrees | rate |")
    w("|---|---|---|---:|---:|---:|")
    for part in PARTS:
        for m, its in sorted(loaded[part].items()):
            marg_m = margin_outcomes(its)
            per_cond: dict[str, list[int]] = {}
            for per_item in marg_m.values():
                for cond, rec in per_item.items():
                    if rec["chosen_is_edited"] is None:
                        continue
                    d = per_cond.setdefault(cond, [0, 0])
                    d[0] += 1
                    d[1] += (rec["margin_after"] > 0) == bool(rec["chosen_is_edited"])
            for cond in sorted(per_cond):
                total_c, agree_c = per_cond[cond]
                rate_c = f"{agree_c / total_c:.4f}" if total_c else "n/a"
                w(f"| {part} | {short_model(m)} | {cond} | {total_c} | {agree_c} | {rate_c} |")
    w("")
    w("Pooled across models, per condition:")
    w("")
    w("| part | condition | rows | probe sign agrees | rate |")
    w("|---|---|---:|---:|---:|")
    pooled_cond_rates: dict[str, dict[str, float | None]] = {}
    for part in PARTS:
        per_cond_pooled: dict[str, list[int]] = {}
        for per_item in margins_by_part[part].values():
            for cond, rec in per_item.items():
                if rec["chosen_is_edited"] is None:
                    continue
                d = per_cond_pooled.setdefault(cond, [0, 0])
                d[0] += 1
                d[1] += (rec["margin_after"] > 0) == bool(rec["chosen_is_edited"])
        pooled_cond_rates[part] = {}
        for cond in ("R1", "R2", "R3", "R4"):
            if cond not in per_cond_pooled:
                continue
            total_c, agree_c = per_cond_pooled[cond]
            rate_c = agree_c / total_c if total_c else None
            pooled_cond_rates[part][cond] = rate_c
            w(f"| {part} | {cond} | {total_c} | {agree_c} | {'n/a' if rate_c is None else f'{rate_c:.4f}'} |")
    w("")
    for part in PARTS:
        rates = pooled_cond_rates[part]
        seq = [rates.get(c) for c in ("R1", "R2", "R3", "R4")]
        if any(r is None for r in seq):
            w(f"Part {part}: at least one condition has no readable rows; monotonicity not "
              f"evaluated.")
            continue
        monotone = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
        w(f"Part {part}: R1→R2→R3→R4 = "
          f"{', '.join(f'{r:.4f}' for r in seq)} -- "
          f"{'monotonically non-decreasing (worst at R1, best at R4)' if monotone else 'not monotonic'}.")
    w("")
    w("R1 and R2 both edit the rival the model just named and rejected; R3 and R4 edit a third")
    w("option it said nothing about. If agreement is systematically lower at the location the")
    w("model actually named, the two instruments have more room to differ exactly where the")
    w("R1−R3 and R2−R4 location contrasts need them to agree -- an asymmetry worth naming as an")
    w("observation, without speculating beyond it: more probability movement near a decision")
    w("boundary gives an argmax read and a raw-probability read more room to disagree than")
    w("either does far from one.")
    w("")
    w("Then the contested contrast. R2−R4 is where the paper reports the two measures pointing")
    w("opposite ways with both intervals excluding zero in Part C.")
    w("")
    w("| part | argmax b, c (favours) | Δp [95% CI] | Δmargin [95% CI] | do Δmargin and argmax agree? |")
    w("|---|---|---|---|---|")
    for part in PARTS:
        disc = discrete_by_part[part]
        d = discrete_contrast(disc, "R2", "R4")
        cont = {
            item_id: {c: {"delta_p_edited": row.get("delta_p_edited")}
                      for c, row in conds.items()}
            for m, its in loaded[part].items() for item_id, conds in
            [(f"{m}::{i}", cc) for i, cc in its.items()]
        }
        cm = paired_mean(cont, "R2", "R4", "delta_p_edited")
        mm = paired_mean(margins_by_part[part], "R2", "R4", "delta_margin")
        favours = "R2" if d["b_a_only"] > d["c_b_only"] else "R4"
        if mm["mean"] is None:
            verdict = "n/a"
        else:
            verdict = "yes" if (mm["mean"] > 0) == (favours == "R2") else "no"
        w(f"| {part} | {d['b_a_only']}, {d['c_b_only']} ({favours}) | {_ci(cm)} | {_ci(mm)} "
          f"| {verdict} |")
    w("")
    w("### Decomposing the R2−R4 mean: sign counts, medians, trimmed means")
    w("")
    w("The mean margin moves the same way as the mean probability and still opposes the")
    w("direction of the flip counts. The remaining explanation is arithmetic rather than")
    w("instrumental: McNemar counts sign changes that cross zero, while the bootstrap above")
    w("averages magnitudes, so a minority of large moves in one direction can outweigh a")
    w("majority of small threshold crossings in the other. This tests that directly, per part,")
    w("per model and pooled: the sign distribution a McNemar-like count would see, the median")
    w("(less sensitive to a few large moves than the mean), and a symmetrically trimmed mean")
    w("(drop the top and bottom 5% / 10%, then average what remains).")
    w("")
    for measure_label, field, is_margin in (
        ("Δmargin", "delta_margin", True),
        ("Δp (raw p_edited delta)", "delta_p_edited", False),
    ):
        w(f"**{measure_label} (R2−R4 per item; positive favours R2)**")
        w("")
        w("| part | scope | n | + | − | 0 | mean [95% CI] | median [95% CI] | trim 5% [95% CI] "
          "| trim 10% [95% CI] |")
        w("|---|---|---:|---:|---:|---:|---|---|---|---|")
        for part in PARTS:
            models = loaded[part]
            pooled_items = {f"{m}::{i}": conds for m, its in models.items()
                            for i, conds in its.items()}
            scopes = [(short_model(m), its) for m, its in sorted(models.items())]
            scopes.append(("**pooled**", pooled_items))
            for scope, its in scopes:
                if is_margin:
                    outcomes = margin_outcomes(its)
                else:
                    outcomes = {
                        item_id: {c: {"delta_p_edited": row.get("delta_p_edited")}
                                  for c, row in conds.items()}
                        for item_id, conds in its.items()
                    }
                diffs = paired_diffs(outcomes, "R2", "R4", field)
                pos, neg, zero = _sign_counts(diffs)
                mean_ci = bootstrap_ci_mean_diff(diffs, seed=SEED, n_resamples=RESAMPLES)
                med_ci = bootstrap_stat(diffs, _median, seed=SEED, n_resamples=RESAMPLES)
                t5_ci = bootstrap_stat(diffs, lambda v: _trimmed_mean(v, 0.05),
                                        seed=SEED, n_resamples=RESAMPLES)
                t10_ci = bootstrap_stat(diffs, lambda v: _trimmed_mean(v, 0.10),
                                         seed=SEED, n_resamples=RESAMPLES)
                w(f"| {part} | {scope} | {len(diffs)} | {pos} | {neg} | {zero} "
                  f"| {_ci(mean_ci)} | {_ci_stat(med_ci)} | {_ci_stat(t5_ci)} "
                  f"| {_ci_stat(t10_ci)} |")
        w("")
    w("If the sign counts and the medians agree with the argmax direction while the means do")
    w("not, that discrepancy is the mechanism the third explanation predicted. If they do")
    w("not agree either, the disagreement is not explained by mean-vs-median arithmetic and")
    w("that should be said plainly rather than papered over.")
    w("")
    w("**Verdict, computed directly from the pooled `delta_margin` sign counts above, per part:**")
    w("")
    for part in PARTS:
        d = discrete_contrast(discrete_by_part[part], "R2", "R4")
        argmax_favours = "R2" if d["b_a_only"] > d["c_b_only"] else "R4"
        pooled_diffs = paired_diffs(margins_by_part[part], "R2", "R4", "delta_margin")
        pos, neg, zero = _sign_counts(pooled_diffs)
        med = _median(pooled_diffs)
        sign_favours = "R2" if pos > neg else ("R4" if neg > pos else "tied")
        median_favours = None if med is None else ("R2" if med > 0 else ("R4" if med < 0 else "tied"))
        sign_agrees = sign_favours == argmax_favours
        median_agrees = median_favours == argmax_favours
        if sign_agrees and median_agrees:
            verdict = ("mean-vs-median arithmetic explains the disagreement: the sign count and "
                       "the median both agree with the argmax, only the mean does not.")
        else:
            verdict = ("mean-vs-median arithmetic does NOT explain the disagreement: the sign "
                       "count and/or the median oppose the argmax direction just as the mean "
                       "does, so the mean is not the odd one out here.")
        w(f"Part {part}: argmax favours {argmax_favours} (b={d['b_a_only']}, c={d['c_b_only']}); "
          f"sign count +{pos}/-{neg}/0:{zero} favours {sign_favours} "
          f"({'agrees' if sign_agrees else 'disagrees'}); median = {_fmt(med)} favours "
          f"{median_favours} ({'agrees' if median_agrees else 'disagrees'}). **{verdict}**")
    w("")
    w("### Discordant vs. concordant items: same population, or different ones?")
    w("")
    w("The sign counts above still oppose McNemar's own direction in Part C, which rules out")
    w("mean-vs-median arithmetic as the explanation: a sign count is exactly the kind of")
    w("majority-rules statistic that was supposed to agree with McNemar if the disagreement were")
    w("only about outliers dominating a mean. It does not. The remaining candidate is that")
    w("McNemar and a plain sign count are not reading the same population at all: McNemar's `b`")
    w("and `c` are counted only from the items where `chosen_is_edited` actually differs between")
    w("R2 and R4 -- the **discordant** items -- while a sign count over every item with a")
    w("reading also includes the far larger **concordant** remainder. This splits R2−R4 into")
    w("exactly those two subsets and reports the sign counts, mean and median of both")
    w("`delta_margin` and `delta_p_edited` on each, per part, per model and pooled.")
    w("")
    pooled_margin_split: dict[str, tuple[list[float], list[float]]] = {}
    for measure_label, field, is_margin in (
        ("Δmargin", "delta_margin", True),
        ("Δp (raw p_edited delta)", "delta_p_edited", False),
    ):
        w(f"**{measure_label}, discordant vs. concordant (R2−R4 per item; positive favours R2)**")
        w("")
        w("| part | scope | subset | n | + | − | 0 | mean [95% CI] | median [95% CI] |")
        w("|---|---|---|---:|---:|---:|---:|---|---|")
        for part in PARTS:
            models = loaded[part]
            pooled_items = {f"{m}::{i}": conds for m, its in models.items()
                            for i, conds in its.items()}
            scopes = [(short_model(m), its) for m, its in sorted(models.items())]
            scopes.append(("**pooled**", pooled_items))
            for scope, its in scopes:
                disc = discrete_outcomes(its)
                if is_margin:
                    outcomes = margin_outcomes(its)
                else:
                    outcomes = {
                        item_id: {c: {"delta_p_edited": row.get("delta_p_edited")}
                                  for c, row in conds.items()}
                        for item_id, conds in its.items()
                    }
                discordant, concordant = discordant_concordant_diffs(
                    disc, outcomes, "R2", "R4", field
                )
                if is_margin and scope == "**pooled**":
                    pooled_margin_split[part] = (discordant, concordant)
                for subset_label, diffs in (("discordant", discordant), ("concordant", concordant)):
                    pos, neg, zero = _sign_counts(diffs)
                    mean_ci = bootstrap_ci_mean_diff(diffs, seed=SEED, n_resamples=RESAMPLES)
                    med_ci = bootstrap_stat(diffs, _median, seed=SEED, n_resamples=RESAMPLES)
                    w(f"| {part} | {scope} | {subset_label} | {len(diffs)} | {pos} | {neg} "
                      f"| {zero} | {_ci(mean_ci)} | {_ci_stat(med_ci)} |")
        w("")
    w("**Verdict, computed directly from the pooled `delta_margin` rows above, per part.**")
    w("The prediction under test was that the discordant subset favours R2 (agreeing with")
    w("McNemar) while the concordant subset favours R4 and is numerous enough to dominate the")
    w("pooled mean -- a population mismatch rather than a real disagreement. The favour each")
    w("subset reports below is read off its **sign count** (a majority vote, the statistic")
    w("least sensitive to the handful of outsized items that can dominate a mean -- see the")
    w("per-model rows above, several of which are one or two items with a delta_margin an order")
    w("of magnitude larger than the rest of that scope's data), not off its mean; where the two")
    w("disagree within a single subset, that is reported explicitly rather than silently")
    w("resolved by picking whichever one to quote.")
    w("")
    for part in PARTS:
        if part not in pooled_margin_split:
            w(f"Part {part}: no pooled `delta_margin` data.")
            continue
        discordant, concordant = pooled_margin_split[part]
        d_pos, d_neg, d_zero = _sign_counts(discordant)
        c_pos, c_neg, c_zero = _sign_counts(concordant)
        d_mean = sum(discordant) / len(discordant) if discordant else None
        c_mean = sum(concordant) / len(concordant) if concordant else None
        d_mean_favours = None if d_mean is None else ("R2" if d_mean > 0 else "R4")
        c_mean_favours = None if c_mean is None else ("R2" if c_mean > 0 else "R4")
        d_favours = "R2" if d_pos > d_neg else ("R4" if d_neg > d_pos else "tied")
        c_favours = "R2" if c_pos > c_neg else ("R4" if c_neg > c_pos else "tied")
        note = ""
        if d_favours != d_mean_favours:
            note = (f" (its mean favours {d_mean_favours} instead -- a handful of outsized "
                     "items dominate the discordant mean too, the same outlier-sensitivity "
                     "problem found in the whole-population mean, recurring inside this "
                     "smaller subset.)")
        if d_favours == "R2" and c_favours == "R4":
            verdict = ("population mismatch confirmed by sign count: the discordant subset's "
                       "majority favours R2, matching McNemar, and the concordant subset's "
                       "majority favours R4, dominating the pooled sign count and mean.")
        elif d_favours == "R4":
            verdict = ("population mismatch NOT confirmed by sign count -- a majority of the "
                       "discordant subset itself favours R4, i.e. on most of exactly the items "
                       "where the generated answer flips, the separately-measured probe's own "
                       "margin moved the *other* way. The two instruments disagree even on the "
                       "population McNemar is built from; McNemar's b/c count (a free-text "
                       "argmax fact) and this sign count (a forced-probe fact) are two different "
                       "measurements of the same items, and they point opposite ways more often "
                       "than not. This is a more serious finding than a population mismatch and "
                       "is reported as such, not smoothed into the mismatch story.")
        else:
            verdict = "tied or inconclusive by sign count; inspect the table above directly."
        w(f"Part {part}: discordant n={len(discordant)} (+{d_pos}/-{d_neg}/0:{d_zero}, sign "
          f"count favours {d_favours}){note}; concordant n={len(concordant)} "
          f"(+{c_pos}/-{c_neg}/0:{c_zero}, sign count favours {c_favours}). **{verdict}**")
    w("")

    # ------------------------------------------------------------- 3. roster interaction
    w("## 3. The roster interaction: is Part B's null a different effect or a smaller sample?")
    w("")
    w("Part A against Part B, on the discrete measure, for the two content contrasts. The")
    w("estimate is the difference between two independent paired risk differences, each part")
    w("resampled within itself. Power is the exact power an experiment of Part B's size has")
    w("against Part A's own observed discordance rate and odds ratio.")
    w("")
    w("| contrast | A: RD [95% CI] | B: RD [95% CI] | A−B [95% CI] | power of B for A's effect |")
    w("|---|---|---|---|---:|")
    for a, b, _why in CONTRASTS[:2]:
        da = discrete_diffs(discrete_by_part["A"], a, b)
        db = discrete_diffs(discrete_by_part["B"], a, b)
        rda = bootstrap_ci_mean_diff(da, seed=SEED, n_resamples=RESAMPLES)
        rdb = bootstrap_ci_mean_diff(db, seed=SEED, n_resamples=RESAMPLES)
        diff = independent_difference(da, db, seed=SEED)
        ca = discrete_contrast(discrete_by_part["A"], a, b)
        n_a = ca["n_paired"]
        disc_rate = (ca["b_a_only"] + ca["c_b_only"]) / n_a if n_a else 0.0
        odds = ca["or"]["or"]
        n_b = discrete_contrast(discrete_by_part["B"], a, b)["n_paired"]
        pw = mcnemar_power(n_b, disc_rate, odds)
        w(f"| {a}−{b} | {_ci(rda, 3)} | {_ci(rdb, 3)} | {_ci(diff, 3)} | {pw:.3f} |")
    w("")

    # -------------------------------------------------------------------- 4. multiplicity
    w("## 4. Multiplicity")
    w("")
    w("Declared primary endpoint: the discrete R3−R4 content contrast in Part A, the contrast")
    w("the design was rebuilt around and the only one fixed before this run's data existed.")
    w("Everything else is secondary. The family below is the twelve discrete tests, four")
    w("contrasts by three parts, Holm-adjusted.")
    w("")
    raw = {}
    for part in PARTS:
        for a, b, _why in CONTRASTS:
            raw[f"{part} {a}−{b}"] = discrete_contrast(discrete_by_part[part], a, b)["p_exact_two_sided"]
    adj = holm(raw)
    w("| test | exact p | Holm-adjusted p | survives at 0.05 |")
    w("|---|---:|---:|---|")
    for key in sorted(raw, key=lambda k: raw[k]):
        w(f"| {key} | {raw[key]:.4f} | {adj[key]:.4f} | {'yes' if adj[key] <= 0.05 else 'no'} |")
    w("")
    w("The continuous family is twelve intervals rather than twelve p-values, so it is reported")
    w("at a Bonferroni-adjusted level instead: a 99.58% interval, which is 1 − 0.05/12.")
    w("")

    # --------------------------------------------------------- 5. the two misstated figures
    w("## 5. Two figures the paper states on the wrong population")
    w("")
    w("### Gate-eight selection skew, per part")
    w("")
    w("The paper reports 62.7% against a 47.4% baseline and describes it as holding in every")
    w("part. Those are run 2's numbers, on its own 295-item built set.")
    w("")
    w("**Join rule.** A stage-2 row enters the attempted pool only if (a) a stage-1 row exists")
    w("for the same `(model, item_id)`, and (b) that stage-1 row's `choice_correct` is not")
    w("`None`. `choice_correct` is `None` exactly when stage 1's free-text `choice` could not be")
    w("parsed -- there is no correctness signal for that item, not a negative one -- and an")
    w("earlier version of this table coerced that `None` to `False` via a bare `bool(...)`,")
    w("silently counting unreadable responses as incorrect and inflating the attempted-pool")
    w("denominator with rows that were never judged. `unjoined (no stage-1 row)` and `unjoined")
    w("(unparseable stage-1 choice)` below are the two ways a stage-2 row can fail this join,")
    w("counted separately.")
    w("")
    w("| part | built correct at stage one | attempted pool | skew | unjoined (no stage-1 row) "
      "| unjoined (unparseable stage-1 choice) |")
    w("|---|---|---|---|---:|---:|")
    for part in PARTS:
        g = gate_eight_skew(results, part)
        if g["built_n"] == 0:
            w(f"| {part} | no join | no join | n/a | {g['unjoined_missing_key']} "
              f"| {g['unjoined_unparseable_choice']} |")
            continue
        w(f"| {part} | {g['built_correct']}/{g['built_n']} = {g['built_rate']:.3f} "
          f"| {g['attempted_correct']}/{g['attempted_n']} = {g['attempted_rate']:.3f} "
          f"| {g['built_rate'] - g['attempted_rate']:+.3f} | {g['unjoined_missing_key']} "
          f"| {g['unjoined_unparseable_choice']} |")
    w("")
    w("### R0 baseline ratio between the two edit locations")
    w("")
    w('The paper says the rival starts "near twice" the third option, in every part. Two')
    w("populations answer this, and they agree in Part B but not in Parts A and C -- see")
    w("`baseline_by_target_paper` and `baseline_by_target_strict` for the exact join each uses.")
    w("")
    w("**Paper population** -- R0's own probe read for the target letter, gated only on that")
    w("letter appearing in R0's own `probs`. This is the population Table 3 and")
    w("the prior run-3 audit's PART 1 used, and this table now reproduces it exactly.")
    w("")
    w("| part | rival n | rival mean | third n | third mean | ratio |")
    w("|---|---:|---:|---:|---:|---:|")
    for part in PARTS:
        pooled = {f"{m}::{i}": conds for m, its in loaded[part].items() for i, conds in its.items()}
        bb = baseline_by_target_paper(pooled)
        ratio = f"{bb['ratio']:.2f}" if bb["ratio"] else "n/a"
        w(f"| {part} | {bb['rival']['n']} | {bb['rival']['mean']:.4f} | {bb['third']['n']} "
          f"| {bb['third']['mean']:.4f} | {ratio} |")
    w("")
    w("**Strict population** -- the run's own `r0_p_target` field, which additionally requires")
    w("the R1/R3 row's own probe to be complete, not just R0's. It coincides with the paper")
    w("population in Part B (no model there is ever partially complete) and diverges from it in")
    w("Parts A and C, where Mistral's probe completeness is well below 100%. This is the")
    w("population internally consistent with the `delta_p_edited` / `delta_margin` columns")
    w("used elsewhere in this report, but it is not what Table 3 reports.")
    w("")
    w("| part | rival n | rival mean | third n | third mean | ratio |")
    w("|---|---:|---:|---:|---:|---:|")
    for part in PARTS:
        pooled = {f"{m}::{i}": conds for m, its in loaded[part].items() for i, conds in its.items()}
        bb = baseline_by_target_strict(pooled)
        ratio = f"{bb['ratio']:.2f}" if bb["ratio"] else "n/a"
        w(f"| {part} | {bb['rival']['n']} | {bb['rival']['mean']:.4f} | {bb['third']['n']} "
          f"| {bb['third']['mean']:.4f} | {ratio} |")
    w("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------- self-check


def self_check() -> int:
    """Arithmetic checks against values computed by hand, no data files needed."""
    problems = []

    o = odds_ratio(36, 15)
    if not (abs(o["or"] - 2.4) < 1e-9 and abs(o["lo"] - 1.31) < 0.01 and abs(o["hi"] - 4.38) < 0.01):
        problems.append(f"odds_ratio(36,15) gave {o}, wanted 2.40 [1.31, 4.38]")

    fixture = {
        "i1": {"R1": True, "R2": False},
        "i2": {"R1": True, "R2": False},
        "i3": {"R1": False, "R2": True},
        "i4": {"R1": True, "R2": True},
    }
    mc = mcnemar(fixture, "R1", "R2")
    if (mc["b_a_only"], mc["c_b_only"]) != (2, 1):
        problems.append(f"mcnemar fixture gave b,c = {mc['b_a_only']},{mc['c_b_only']}, wanted 2,1")

    # Holm on a known family: three p-values, m=3, step-down and monotone.
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    if not (abs(adj["a"] - 0.03) < 1e-12 and abs(adj["c"] - 0.06) < 1e-12
            and abs(adj["b"] - 0.06) < 1e-12):
        problems.append(f"holm gave {adj}, wanted a=0.03, c=0.06, b=0.06")

    # A margin is the target minus its best competitor, and its sign is what an argmax reads.
    if _margin({"A": 0.5, "B": 0.3, "C": 0.2}, "A") != 0.2:
        problems.append("margin of a winning letter is wrong")
    if _margin({"A": 0.3, "B": 0.5, "C": 0.2}, "A") != -0.2:
        problems.append("margin of a losing letter is wrong")

    # Power is a probability, rises with n, and a null effect sits near alpha or below.
    p_small = mcnemar_power(50, 0.10, 3.22)
    p_large = mcnemar_power(400, 0.10, 3.22)
    if not 0.0 <= p_small <= p_large <= 1.0:
        problems.append(f"power not monotone in n: {p_small} then {p_large}")
    if mcnemar_power(300, 0.10, 1.0) > 0.05:
        problems.append("power at a null odds ratio exceeds alpha")

    # Median of an odd- and an even-length sample, by hand.
    if _median([1.0, 3.0, 2.0]) != 2.0:
        problems.append("median of an odd-length sample is wrong")
    if _median([1.0, 2.0, 3.0, 4.0]) != 2.5:
        problems.append("median of an even-length sample is wrong")

    # A symmetric 10% trim on ten sorted values drops the one smallest and one largest.
    if _trimmed_mean([float(i) for i in range(10)], 0.10) != sum(range(1, 9)) / 8:
        problems.append("10% trimmed mean of range(10) is wrong")

    # Sign counts: 2 positive, 1 negative, 1 exactly zero.
    if _sign_counts([0.1, -0.2, 0.0, 0.3]) != (2, 1, 1):
        problems.append("sign counts on a hand-picked sample are wrong")

    for problem in problems:
        print("FAIL", problem)
    print("self-check:", "ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", type=Path,
                    help="the results tree holding exp3a, exp3b, exp3c (default: results)")
    ap.add_argument("--out", default=None, type=Path,
                    help="where to write the report (default: a fixed location outside code/)")
    ap.add_argument("--self-check", action="store_true", help="arithmetic checks only")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()

    out = args.out or (REPO_ROOT / "code" / "analysis_run3_report.txt")
    report = build_report(args.results)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
