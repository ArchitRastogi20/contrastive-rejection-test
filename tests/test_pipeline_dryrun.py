"""End-to-end smoke test of harness.run_pilot's dry-run mode with a stub model: no GPU, no network."""

import json

from harness.run_pilot import main


def test_dry_run_writes_a_summary_and_raw_records(tmp_path):
    rc = main(["--dry-run", "--limit", "3", "--out-dir", str(tmp_path)])
    assert rc == 0

    summary = json.loads((tmp_path / "pilot_summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["n_items"] == 3
    assert summary["seed"] and summary["temperature"] == 0.0
    assert "UTC" in summary["finished_utc"]
    assert set(summary["per_run"]) == {"stub|spontaneous", "stub|elicited"}

    for arm in ("spontaneous", "elicited"):
        lines = (tmp_path / f"raw_stub_{arm}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        rec = json.loads(lines[0])
        assert rec["arm"] == arm
        assert rec["response"]  # raw output is always kept
        assert rec["gold_letter"] in "ABCD"
        assert "rejections" in rec


def test_dry_run_finds_the_rejections_the_stub_states(tmp_path):
    main(["--dry-run", "--limit", "3", "--out-dir", str(tmp_path)])
    counts = json.loads((tmp_path / "pilot_summary.json").read_text(encoding="utf-8"))
    run = counts["per_run"]["stub|spontaneous"]

    # the stub cycles three replies: two name a concrete missing attribute, one is vague
    assert run["items"] == 3
    assert run["with_rejection"] == 2
    assert run["specific"] == 2
    assert run["usable"] >= 1
    assert run["abort_reason"] is None


def test_heartbeat_is_written(tmp_path):
    main(["--dry-run", "--limit", "3", "--out-dir", str(tmp_path)])
    beat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["total"] == 3
    assert "UTC" in beat["updated_utc"]
    assert beat["abort_reason"] is None
    assert beat["vram_strikes"] == 0
