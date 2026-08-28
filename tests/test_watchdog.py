"""Tests for harness.watchdog: VRAM-breach abort logic and GPU-seconds accounting."""

import harness.watchdog as W
from harness.watchdog import Watchdog, commit_gpu_seconds, cumulative_gpu_seconds


def test_vram_breach_aborts_only_after_consecutive_strikes(monkeypatch):
    readings = iter([0.80, 0.97, 0.80, 0.97, 0.98, 0.99])
    monkeypatch.setattr(W, "vram_fraction", lambda: next(readings))
    dog = Watchdog(label="t", total=6, budget_min=60)

    dog.sample_vram()                      # 0.80
    dog.sample_vram()                      # 0.97, strike 1
    assert dog.should_abort is False
    dog.sample_vram()                      # 0.80 -- the streak resets
    assert dog.vram_strikes == 0
    dog.sample_vram()                      # 0.97
    dog.sample_vram()                      # 0.98
    assert dog.should_abort is False       # two strikes is not three
    dog.sample_vram()                      # 0.99, strike 3
    assert dog.should_abort is True
    assert "VRAM" in dog.abort_reason
    assert dog.vram_peak == 0.99


def test_no_cuda_means_no_guard_and_no_crash(monkeypatch):
    monkeypatch.setattr(W, "vram_fraction", lambda: None)
    dog = Watchdog(label="t", total=1, budget_min=60)
    assert dog.sample_vram() is None
    assert dog.should_abort is False


def test_wall_clock_budget_aborts(monkeypatch):
    dog = Watchdog(label="t", total=10, budget_min=1)
    dog.started -= 120  # pretend two minutes have passed
    assert dog.should_abort is True
    assert "wall-clock" in dog.abort_reason


def test_gpu_ledger_appends_and_accumulates(tmp_path):
    path = tmp_path / "gpu_ledger.csv"
    assert cumulative_gpu_seconds(path) == 0.0

    total = commit_gpu_seconds("a/spontaneous", "m", "vllm", 40, 120.4, 0.83, None, path)
    assert total == 120.4

    # a crashed run still records what it spent, which is the whole point of the ledger
    total = commit_gpu_seconds("a/elicited", "m", "vllm", 7, 30.0, 0.96, "VRAM", path)
    assert total == 150.4

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3  # header plus two entries
    assert "VRAM" in rows[-1]
    assert "+00:00" in rows[-1]  # timestamps carry the UTC offset


def test_dtype_follows_compute_capability():
    from harness.models import preferred_dtype

    # a T4 is 7.5 and has no bfloat16 at all; vLLM refuses to load if asked for it
    assert preferred_dtype((7, 5)) == "float16"
    assert preferred_dtype((8, 6)) == "bfloat16"
    assert preferred_dtype((9, 0)) == "bfloat16"
    assert preferred_dtype(None) in ("float16", "bfloat16")  # asks the real device


def test_profile_follows_vram(monkeypatch):
    import harness.models as M
    from harness.config import MODEL_PROFILES

    monkeypatch.setattr(M, "device_vram_gib", lambda: 15.7)   # a T4
    assert M.resolve_profile("auto") == "t4"
    monkeypatch.setattr(M, "device_vram_gib", lambda: 23.6)   # a 3090 Ti
    assert M.resolve_profile("auto") == "3090ti"
    monkeypatch.setattr(M, "device_vram_gib", lambda: None)   # no device
    assert M.resolve_profile("auto") == "3090ti"
    assert M.resolve_profile("t4") == "t4"                    # explicit wins

    for name, roster in MODEL_PROFILES.items():
        assert len(roster) == 3, name
