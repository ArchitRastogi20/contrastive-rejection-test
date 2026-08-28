"""Tests for the round-6 support code: harness.analyze_necessity's arithmetic, and a syntax check
on scripts/monitor.sh and scripts/prefetch_round6.sh.

Hand-written fixtures only; no GPU, no network, seconds to run.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from harness.analyze_necessity import (
    ASSUMED_MODEL_LOAD_S,
    E1_BUILT_YIELD_4OPT,
    _rate_from_ledger,
    _verify_ledger_header,
    build_nested,
    interpretable,
    project_e1_cost,
    report_discrete,
    report_integrity,
    report_strata,
)
from harness.config import CODE_ROOT
from harness.watchdog import LEDGER_FIELDS


# ------------------------------------------------------------------- fixture rows (shared shape)


def _row(model, item_id, condition, *, still, gold=True, relevant=False, delta=None):
    return {
        "model": model, "part": "A", "item_id": item_id, "condition": condition,
        "attribute": "director", "rival_letter": "B", "rival_title": "Rival",
        "chosen_letter": "A", "chosen_is_gold": gold, "question_relevant_attribute": relevant,
        "choice": "A" if still else "B", "still_chooses_original": still,
        "letter_probe": {}, "p_original": None, "r0_p_target": None,
        "delta_p_original": delta, "response": "stub",
    }


def _known_b_c_rows():
    """One model, N0 always True (nothing here tests integrity). N1 vs N2 engineered to a known
    b=3 (N1 True, N2 False), c=1 (N1 False, N2 True) so odds_ratio/mcnemar are hand-checkable:
    b/c = 3, matched-pairs OR = 3.0."""
    rows = []
    # b: N1=True, N2=False (three items)
    for i in range(3):
        item_id = f"b{i}"
        rows.append(_row("M", item_id, "N0", still=True))
        rows.append(_row("M", item_id, "N1", still=True))
        rows.append(_row("M", item_id, "N2", still=False))
    # c: N1=False, N2=True (one item)
    rows.append(_row("M", "c0", "N0", still=True))
    rows.append(_row("M", "c0", "N1", still=False))
    rows.append(_row("M", "c0", "N2", still=True))
    # concordant filler, contributes to n but not b/c
    rows.append(_row("M", "x0", "N0", still=True))
    rows.append(_row("M", "x0", "N1", still=True))
    rows.append(_row("M", "x0", "N2", still=True))
    return rows


# --------------------------------------------------------------------------------- discrete/OR


def test_discrete_contrast_known_b_c():
    nested = build_nested(_known_b_c_rows())
    items = interpretable(nested)
    _, payload = report_discrete(nested)
    d = payload["N1-N2"]["**pooled**"]
    assert (d["b_a_only"], d["c_b_only"]) == (3, 1)
    assert d["or"]["or"] == pytest.approx(3.0)
    assert d["n_paired"] == 5
    assert items  # sanity: interpretable() kept everything (no N0 flips in this fixture)


def test_discrete_contrast_matches_hand_mcnemar_exact_p():
    """b=3, c=1, n=4 discordant: exact two-sided binomial p = 2 * P(X<=1) for X~Binomial(4,0.5)
    = 2 * (1 + 4)/16 = 0.625."""
    nested = build_nested(_known_b_c_rows())
    _, payload = report_discrete(nested)
    d = payload["N1-N2"]["**pooled**"]
    assert d["p_exact_two_sided"] == pytest.approx(0.625)


# --------------------------------------------------------------------------------- N0 flip counting


def test_integrity_counts_n0_flips_and_pools_them():
    rows = [
        _row("M1", "i1", "N0", still=True), _row("M1", "i1", "N1", still=True),
        _row("M1", "i1", "N2", still=True),
        _row("M1", "i2", "N0", still=False), _row("M1", "i2", "N1", still=True),
        _row("M1", "i2", "N2", still=True),
        _row("M2", "i3", "N0", still=False), _row("M2", "i3", "N1", still=True),
        _row("M2", "i3", "N2", still=True),
    ]
    nested = build_nested(rows)
    _, payload = report_integrity(nested)
    assert payload["per_model"]["M1"] == {
        "n": 2, "reproduced": 1, "flipped": 1, "unreadable": 0, "flip_rate": pytest.approx(0.5),
    }
    assert payload["per_model"]["M2"]["flipped"] == 1
    assert payload["pooled"]["n"] == 3
    assert payload["pooled"]["flipped"] == 2
    assert payload["pooled"]["reproduced"] == 1


def test_interpretable_excludes_n0_flips_and_unreadable():
    rows = [
        _row("M", "flip", "N0", still=False), _row("M", "flip", "N1", still=True),
        _row("M", "flip", "N2", still=True),
        _row("M", "clean", "N0", still=True), _row("M", "clean", "N1", still=True),
        _row("M", "clean", "N2", still=True),
    ]
    # an N0 row with an unreadable (None) choice
    unreadable = _row("M", "unread", "N0", still=True)
    unreadable["still_chooses_original"] = None
    rows.append(unreadable)
    rows.append(_row("M", "unread", "N1", still=True))
    rows.append(_row("M", "unread", "N2", still=True))

    nested = build_nested(rows)
    kept = interpretable(nested)
    assert set(kept["M"]) == {"clean"}


# --------------------------------------------------------------------------------- strata split


def test_strata_routes_rows_by_chosen_is_gold_and_relevance():
    rows = [
        _row("M", "g1", "N0", still=True, gold=True, relevant=False),
        _row("M", "g1", "N1", still=False, gold=True, relevant=False),
        _row("M", "g1", "N2", still=True, gold=True, relevant=False),

        _row("M", "ng1", "N0", still=True, gold=False, relevant=True),
        _row("M", "ng1", "N1", still=False, gold=False, relevant=True),
        _row("M", "ng1", "N2", still=True, gold=False, relevant=True),

        _row("M", "ng2", "N0", still=True, gold=False, relevant=False),
        _row("M", "ng2", "N1", still=True, gold=False, relevant=False),
        _row("M", "ng2", "N2", still=True, gold=False, relevant=False),
    ]
    nested = build_nested(rows)
    _, payload = report_strata(nested)

    gold_strat = payload["chosen_is_gold"]["N1-N2"]
    assert gold_strat["gold"]["n_rows"] == 1
    assert gold_strat["not_gold"]["n_rows"] == 2

    rel_strat = payload["question_relevant_attribute"]["N1-N2"]
    assert rel_strat["relevant"]["n_rows"] == 1
    assert rel_strat["not_relevant"]["n_rows"] == 2

    # g1 is the only discordant-favouring-N2 item in the gold bucket: b=0, c=1
    gold_disc = gold_strat["gold"]["discrete"]
    assert (gold_disc["b_a_only"], gold_disc["c_b_only"]) == (0, 1)


def test_strata_flags_a_too_small_stratum_in_the_report_text():
    """A single-row stratum (fewer than MIN_DISCORDANT_FOR_INFERENCE discordant pairs) must be
    marked in the rendered table, not silently reported as if it supported inference."""
    rows = [
        _row("M", "only", "N0", still=True, gold=True),
        _row("M", "only", "N1", still=False, gold=True),
        _row("M", "only", "N2", still=True, gold=True),
    ]
    nested = build_nested(rows)
    text, _payload = report_strata(nested)
    assert "yes (no rows)" in text or "| yes |" in text


# ------------------------------------------------------------------- continuous field is None


def test_continuous_outcomes_skip_rows_with_a_missing_delta():
    from harness.analyze_necessity import report_continuous

    rows = [
        _row("M", "has_delta", "N0", still=True, delta=None),
        _row("M", "has_delta", "N1", still=True, delta=-0.2),
        _row("M", "has_delta", "N2", still=True, delta=0.1),

        _row("M", "missing_delta", "N0", still=True, delta=None),
        _row("M", "missing_delta", "N1", still=True, delta=None),  # probe never completed
        _row("M", "missing_delta", "N2", still=True, delta=0.05),
    ]
    nested = build_nested(rows)
    _, payload = report_continuous(nested)
    cm = payload["N1-N2"]["**pooled**"]
    assert cm["n"] == 1  # only has_delta is complete on both sides
    assert cm["mean"] == pytest.approx(-0.3)


def test_continuous_report_does_not_raise_when_every_row_is_missing_a_delta():
    from harness.analyze_necessity import report_continuous

    rows = [
        _row("M", "i1", "N0", still=True, delta=None),
        _row("M", "i1", "N1", still=True, delta=None),
        _row("M", "i1", "N2", still=True, delta=None),
    ]
    nested = build_nested(rows)
    text, payload = report_continuous(nested)
    assert payload["N1-N2"]["**pooled**"]["n"] == 0
    assert "n/a" in text


# --------------------------------------------------------------------------------- E1 estimate


def test_rate_from_ledger_averages_only_matching_vllm_rows():
    ledger = [
        {"label": "A/stage1", "backend": "vllm", "items": "10", "gpu_seconds": "20.0"},
        {"label": "B/stage1", "backend": "vllm", "items": "10", "gpu_seconds": "40.0"},
        {"label": "C/stage1", "backend": "transformers", "items": "10", "gpu_seconds": "999.0"},
        {"label": "D/other", "backend": "vllm", "items": "10", "gpu_seconds": "999.0"},
    ]
    rate, n_rows, total_s = _rate_from_ledger(ledger, "/stage1")
    assert rate == pytest.approx(3.0)  # (20+40)/(10+10)
    assert n_rows == 2
    assert total_s == pytest.approx(60.0)


def test_rate_from_ledger_returns_none_on_no_match():
    rate, n_rows, total_s = _rate_from_ledger([], "/stage1")
    assert rate is None
    assert n_rows == 0
    assert total_s == 0.0


def test_project_e1_cost_matches_hand_computation():
    ledger = [
        {"label": "A/stage1", "backend": "vllm", "items": "100", "gpu_seconds": "250.0"},
        {"label": "A/probe_variants", "backend": "vllm", "items": "500", "gpu_seconds": "50.0"},
    ]
    proj = project_e1_cost(
        ledger, cumulative_gpu_s=500.0, built_low=300, built_high=300,  # low==high: one figure
        yield_rate=0.5, checkpoints=2, conditions=4, load_overhead_s=10.0, budget_s=100_000.0,
    )
    assert proj["reask_rate_s_per_item"] == pytest.approx(2.5)
    assert proj["probe_rate_s_per_row"] == pytest.approx(0.1)
    # attempted = ceil(300/0.5) = 600; stage1 = 600*2.5 = 1500
    # per_condition_s = 2.5+0.1 = 2.6 (no /necessity rows); stage3 = 300*4*2.6 = 3120
    # per-checkpoint total = 1500+3120+10 = 4630; x2 checkpoints = 9260
    assert proj["per_checkpoint_low"]["attempted"] == 600
    assert proj["per_checkpoint_low"]["stage1_s"] == pytest.approx(1500.0)
    assert proj["per_checkpoint_low"]["stage3_s"] == pytest.approx(3120.0)
    assert proj["total_low_s"] == pytest.approx(9260.0)
    assert proj["total_low_s"] == proj["total_high_s"]  # low==high by construction here
    assert proj["remaining_s"] == pytest.approx(99_500.0)
    assert proj["fits_low"] is True


def test_project_e1_cost_prefers_necessity_ledger_rows_when_present():
    ledger = [
        {"label": "A/stage1", "backend": "vllm", "items": "100", "gpu_seconds": "250.0"},
        {"label": "A/probe_variants", "backend": "vllm", "items": "500", "gpu_seconds": "50.0"},
        {"label": "A/necessity", "backend": "vllm", "items": "50", "gpu_seconds": "300.0"},
    ]
    proj = project_e1_cost(
        ledger, cumulative_gpu_s=0.0, built_low=100, built_high=100,
        yield_rate=1.0, checkpoints=1, conditions=3, load_overhead_s=0.0, budget_s=1e9,
    )
    # necessity: 300/50 = 6.0 s/item over 3 (E3) conditions = 2.0 s/condition
    assert proj["per_condition_s"] == pytest.approx(2.0)
    assert proj["necessity_rate_s_per_item"] == pytest.approx(6.0)
    # stage3 = built(100) * conditions(3) * per_condition_s(2.0) = 600
    assert proj["per_checkpoint_low"]["stage3_s"] == pytest.approx(600.0)


def test_project_e1_cost_falls_back_gracefully_on_an_empty_ledger():
    """No round-6 summary and no ledger history: the projection must still return a usable,
    assumption-flagged number rather than raising or dividing by zero."""
    proj = project_e1_cost([], cumulative_gpu_s=0.0)
    assert proj["reask_rate_s_per_item"] is not None
    assert proj["probe_rate_s_per_row"] is not None
    assert proj["total_low_s"] > 0
    assert proj["total_high_s"] >= proj["total_low_s"]
    assert proj["load_overhead_s_per_checkpoint"] == ASSUMED_MODEL_LOAD_S
    assert any("assumed" in a for a in proj["assumptions"])


def test_e1_built_yield_matches_the_documented_run3_figure():
    assert E1_BUILT_YIELD_4OPT == pytest.approx(387 / 1204)


def test_verify_ledger_header_accepts_the_real_schema_and_rejects_a_truncated_one():
    _verify_ledger_header(list(LEDGER_FIELDS))  # must not raise
    with pytest.raises(ValueError):
        _verify_ledger_header(["timestamp_utc", "label"])
    with pytest.raises(ValueError):
        _verify_ledger_header(None)


# --------------------------------------------------------------------------------- shell scripts


def _bash_or_skip() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available on this machine")
    return bash


@pytest.mark.parametrize("script", ["monitor.sh", "prefetch_round6.sh"])
def test_shell_script_passes_bash_syntax_check(script):
    bash = _bash_or_skip()
    path = CODE_ROOT / "scripts" / script
    assert path.exists(), f"missing {path}"
    result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_monitor_sh_usage_error_on_missing_arguments():
    """Cheap, GPU-free behavioural check beyond the syntax check: no PID/RUN_DIR must exit 2
    immediately rather than hanging in the monitor loop."""
    bash = _bash_or_skip()
    path = CODE_ROOT / "scripts" / "monitor.sh"
    result = subprocess.run([bash, str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()
