"""Configuration, paths and UTC logging for the contrastive-rejection pilot."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
RESULTS_DIR = CODE_ROOT / "results"
FIXTURE_DIR = CODE_ROOT / "tests" / "fixtures"
GPU_LEDGER = RESULTS_DIR / "gpu_ledger.csv"

# --------------------------------------------------------------------------------- data

DATASET_ID = os.environ.get("PILOT_DATASET", "framolfese/2WikiMultihopQA")
DATASET_SPLIT = os.environ.get("PILOT_SPLIT", "validation")

# Alternatives, in order, if the primary repo's schema has moved or it disappears. The second
# was loaded successfully in the previous study (full validation scan, 12,576 rows), so it is a
# known-good route rather than a guess -- but its field shapes were not verified for *this*
# design, so run --inspect-schema before trusting it.
DATASET_FALLBACKS = ("voidful/2WikiMultihopQA",)

# `datasets` 5.x dropped legacy script-based loading. When a repo has no parquet export the
# loader retries on this revision, which is how the previous study reached QASPER.
PARQUET_REVISION = "refs/convert/parquet"

# -------------------------------------------------------------------------------- models

# Two rosters, because the card decides what fits. `--profile auto` picks by VRAM.
#
#   3090ti  24 GB, compute capability 8.6: three 7-8B families in bfloat16.
#   t4      16 GB, compute capability 7.5: three 3-4B families in float16. A 7B in float16 is
#           about 15 GB of weights alone and does not leave room for a KV cache, and bfloat16
#           does not exist below capability 8.0 -- vLLM refuses it outright on a T4.
#
# The T4 roster is not a smaller version of the same experiment: a low spontaneous-rejection
# rate at 3B is confounded with capability, so a T4 result is a lower bound on the gate and
# cannot fail it on the models' behalf.
MODEL_PROFILES = {
    "3090ti": [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "t4": [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ],
}

MODELS = MODEL_PROFILES["3090ti"]

# Below this much VRAM, `--profile auto` chooses the small roster.
SMALL_CARD_GIB = 20.0

# Ungated mirrors of the two gated repos, used automatically if a download is refused. Same
# weights, no licence click, so a gated repo never blocks a run.
UNGATED_MIRRORS = {
    "meta-llama/Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "unsloth/mistral-7b-instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
}

# A fourth, smaller, ungated model if a family has to be dropped entirely.
# NOT a Qwen3: vllm 0.6.3.post1 predates that architecture. See requirements.txt.
SUBSTITUTE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# This repo ships a redundant consolidated.safetensors -- the same ~15 GB of weights the
# sharded files already carry. Excluded at download time, per the previous study's ledger.
DOWNLOAD_EXCLUDES = ("consolidated.safetensors",)

TRUST_REMOTE_CODE = os.environ.get("PILOT_TRUST_REMOTE_CODE", "") == "1"

# ------------------------------------------------------------------------------ decoding

SEED = 20260820
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 400

# max_model_len is computed per run from the actual prompts (see run_pilot.compute_max_len)
# rather than fixed: too low silently truncates a profile, too high starves the KV cache and
# causes preemption. These bound that calculation.
MAX_MODEL_LEN_FLOOR = 2048
MAX_MODEL_LEN_CAP = 8192

# ---------------------------------------------------------------------------------- VRAM

# Measured on this card: vLLM's CUDA-graph capture pushes real usage above its own accounting,
# so a 0.90 target was observed at 90.3-93.7% real, and an AWQ run at 0.90 breached outright.
# 0.85 ran clean. It costs some KV-cache blocks and therefore some throughput. Keep it.
VLLM_GPU_MEM_UTILIZATION = float(os.environ.get("PILOT_GPU_MEM_UTIL", "0.85"))

# CUDA-graph capture is what pushes real usage past vLLM's accounting. On a small card the
# graphs are not worth the headroom they cost, so eager mode is the default there.
VLLM_ENFORCE_EAGER = os.environ.get("PILOT_ENFORCE_EAGER", "auto")

# The watchdog aborts a model after this many consecutive samples above the ceiling. This is
# the mechanism that caught the breach above after 150 GPU-seconds instead of after an hour.
VRAM_ABORT_FRACTION = float(os.environ.get("PILOT_VRAM_ABORT", "0.95"))
VRAM_BREACH_STRIKES = 3

# --------------------------------------------------------------------------------- study

N_ITEMS = 40

# The gold option plus (N_OPTIONS - 1) type-matched distractors. `data.build_item`/
# `data.build_items` already take this as an explicit `n_options` argument -- the constant
# here is only the default every caller passes unless told otherwise. Kept at 4 by default so
# runs 1 and 2 stay exactly reproducible: at 4
# options, excluding the rival, the model's stage-1 choice and gold leaves exactly one option
# for repair.build_conditions_with_diagnostics's R3/R4 target, which is why gate 8 (the option
# R3/R4 edits must itself lack the named attribute) was a forced coin flip rather than a real
# choice -- 444 of run 2's 486 integrity-gate drops. A larger value gives that selection more
# than one option to choose from without touching gate 8 itself. Overridable, same convention
# as the VRAM/budget knobs above, so an offline rebuild can be pointed at a different value
# without a code change; the next run is expected to use 6.
N_OPTIONS = int(os.environ.get("PILOT_N_OPTIONS", "4"))

# R3/R4's target is preferred to be an option that already lacks the named attribute, which is
# what relieves gate 8. Active only at option counts strictly above this threshold, so the
# default of 4 reproduces runs 1 and 2 exactly. Set it to 3 to enable the preference at 4
# options: measured on run 2's own elicitations that recovers roughly a third more items
# (Qwen 95 -> 126) with no new generation, because the premise that 4 options force a single
# candidate is false whenever the model's own answer is correct -- 62.7% of built items.
PREFER_LACKING_TARGET_ABOVE = int(os.environ.get("PILOT_PREFER_ABOVE", "4"))
GATE_USABLE_ITEMS = 15  # pre-committed; see the experiment design doc. Do not move it to pass.

# Wall-clock budget per model, in minutes. Exceeding it aborts that model and moves on.
PER_MODEL_BUDGET_MIN = float(os.environ.get("PILOT_MODEL_BUDGET_MIN", "45"))

# Whole-run GPU budget, for the ledger's running total. The project allowance is 20 h.
PROJECT_GPU_BUDGET_S = 72_000


def now_utc() -> datetime:
    return datetime.now(UTC)


def stamp_utc(dt: datetime | None = None) -> str:
    return (dt or now_utc()).strftime("%Y-%m-%d %H:%M:%S UTC")


def iso_utc(dt: datetime | None = None) -> str:
    """ISO-8601 UTC, with the +00:00 offset."""
    return (dt or now_utc()).isoformat(timespec="seconds")


class _UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802 - stdlib signature
        dt = datetime.fromtimestamp(record.created, UTC)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S UTC")


def setup_logging(logfile: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Log to stderr and, optionally, to a file. Every timestamp is UTC and says so."""
    fmt = _UTCFormatter("%(asctime)s %(levelname)-7s %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root
