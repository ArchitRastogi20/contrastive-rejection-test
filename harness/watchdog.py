"""Progress tracking, wall-clock budget, VRAM guard, and the append-only GPU-time ledger."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from .config import (
    GPU_LEDGER,
    PROJECT_GPU_BUDGET_S,
    VRAM_ABORT_FRACTION,
    VRAM_BREACH_STRIKES,
    iso_utc,
    now_utc,
    stamp_utc,
)

log = logging.getLogger(__name__)


def vram_fraction() -> float | None:
    """Fraction of device memory in use, or None when there is no CUDA device to ask."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch means no guard, not a crash
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return (total - free) / total if total else None


@dataclass
class Watchdog:
    label: str
    total: int
    budget_min: float
    heartbeat: Path | None = None
    report_every: int = 5

    done: int = 0
    started: float = field(default_factory=time.monotonic)
    vram_strikes: int = 0
    vram_peak: float = 0.0
    abort_reason: str | None = None

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    @property
    def rate_s(self) -> float:
        return self.elapsed_s / self.done if self.done else 0.0

    @property
    def over_budget(self) -> bool:
        return self.elapsed_s > self.budget_min * 60.0

    @property
    def should_abort(self) -> bool:
        if self.abort_reason:
            return True
        if self.over_budget:
            self.abort_reason = f"over its {self.budget_min:.0f}-minute wall-clock budget"
            return True
        return False

    def eta(self) -> str:
        if not self.done:
            return "unknown"
        remaining = max(0, self.total - self.done) * self.rate_s
        return f"{stamp_utc(now_utc() + timedelta(seconds=remaining))} (in {remaining/60:.1f} min)"

    def sample_vram(self) -> float | None:
        """One VRAM reading. Consecutive readings over the ceiling abort the model."""
        frac = vram_fraction()
        if frac is None:
            return None
        self.vram_peak = max(self.vram_peak, frac)
        if frac > VRAM_ABORT_FRACTION:
            self.vram_strikes += 1
            log.warning(
                "%s: VRAM %.1f%% over the %.0f%% ceiling (strike %d of %d)",
                self.label, frac * 100, VRAM_ABORT_FRACTION * 100,
                self.vram_strikes, VRAM_BREACH_STRIKES,
            )
            if self.vram_strikes >= VRAM_BREACH_STRIKES and not self.abort_reason:
                self.abort_reason = (
                    f"VRAM above {VRAM_ABORT_FRACTION:.0%} for {VRAM_BREACH_STRIKES} "
                    f"consecutive samples, peak {self.vram_peak:.1%}"
                )
        else:
            self.vram_strikes = 0
        return frac

    def tick(self, n: int = 1) -> None:
        self.done += n
        self.sample_vram()
        if self.heartbeat is not None:
            self._write()
        if self.done % self.report_every == 0 or self.done == self.total:
            log.info(
                "%s: %d/%d done, %.1fs/item, elapsed %.1f min, VRAM peak %.1f%%, ETA %s",
                self.label, self.done, self.total, self.rate_s, self.elapsed_s / 60.0,
                self.vram_peak * 100, self.eta(),
            )
        if self.should_abort:
            log.warning("%s: aborting -- %s", self.label, self.abort_reason)

    def _write(self) -> None:
        self.heartbeat.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "elapsed_min": round(self.elapsed_s / 60.0, 2),
            "seconds_per_item": round(self.rate_s, 2),
            "vram_peak_pct": round(self.vram_peak * 100, 1),
            "vram_strikes": self.vram_strikes,
            "eta_utc": self.eta(),
            "updated_utc": stamp_utc(),
            "abort_reason": self.abort_reason,
        }
        tmp = self.heartbeat.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.heartbeat)


LEDGER_FIELDS = ("timestamp_utc", "label", "model", "backend", "items", "gpu_seconds",
                 "vram_peak_pct", "abort_reason")


def commit_gpu_seconds(
    label: str, model: str, backend: str, items: int, seconds: float,
    vram_peak: float = 0.0, abort_reason: str | None = None, path: Path = GPU_LEDGER,
) -> float:
    """Append one row to the GPU ledger and return the new cumulative total.

    Call this from a `finally`. A run that crashes has still spent the time, and a ledger that
    records zero for it is worse than no ledger.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow(
            {
                "timestamp_utc": iso_utc(),
                "label": label,
                "model": model,
                "backend": backend,
                "items": items,
                "gpu_seconds": round(seconds, 1),
                "vram_peak_pct": round(vram_peak * 100, 1),
                "abort_reason": abort_reason or "",
            }
        )

    total = cumulative_gpu_seconds(path)
    log.info(
        "GPU ledger: +%.1fs for %s, cumulative %.1fs (%.2f h, %.1f%% of the %.0f h allowance)",
        seconds, label, total, total / 3600, 100 * total / PROJECT_GPU_BUDGET_S,
        PROJECT_GPU_BUDGET_S / 3600,
    )
    return total


def cumulative_gpu_seconds(path: Path = GPU_LEDGER) -> float:
    if not path.exists():
        return 0.0
    with open(path, newline="", encoding="utf-8") as fh:
        return sum(float(row["gpu_seconds"] or 0) for row in csv.DictReader(fh))
