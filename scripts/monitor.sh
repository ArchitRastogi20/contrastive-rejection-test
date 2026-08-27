#!/usr/bin/env bash
# External health-check loop for a running pilot job on the metered pod.
#
# The in-process watchdog (pilot/watchdog.py) samples VRAM, tracks an ETA, aborts a model after
# three consecutive over-ceiling readings, and writes heartbeat.json -- but it cannot report that
# the *process itself* has died. This script is the outside check: it does not touch the
# watchdog's own logic (no VRAM-abort decision is made here, no GPU-seconds are committed), it
# only observes the PID, the heartbeat file, nvidia-smi and disk, and tells a human when one of
# them looks wrong.
#
#   bash scripts/monitor.sh <PID> <RUN_DIR> [LOG_PATH] > results/monitor.log 2>&1 &
#
# <RUN_DIR> is the --out-dir a run was given (heartbeat.json and *.log live there). LOG_PATH
# overrides the run log to tail; by default the most recently modified *.log under RUN_DIR is
# used, since every pilot entry point names its own log file differently (run_necessity.log,
# probe_variants.log, ...).
#
# Failure signatures this script watches for, and what each means:
#
#   1. heartbeat.json is stale while the monitored PID is still alive -- the run has hung: stuck
#      in a backend call, deadlocked, or the watchdog itself wedged without killing the process.
#      STALE_HEARTBEAT_S is chosen relative to this project's own observed tick rate (worst
#      observed on this project's ledger: ~5.2 s/item, and Watchdog.tick() writes the heartbeat
#      on every tick, i.e. every condition) plus the gap while a model loads between stages or
#      between models, which is real, unlogged wall-clock time and up to a few minutes on this
#      project's own console.log -- generous enough not to false-positive on a slow model load,
#      tight enough to catch a real hang within two or three monitor cycles.
#   2. the monitored PID has exited, no heartbeat is found, or the last heartbeat shows done <
#      total with no `abort_reason` recorded -- a silent crash (OOM-killed, `kill -9`, a Python
#      exception outside the run's own try/finally), as opposed to a run that stopped and said
#      why in its own heartbeat.
#   3. VRAM used/total from `nvidia-smi` is above VRAM_ABORT_PCT for VRAM_STRIKES consecutive
#      checks -- the in-process watchdog samples far more often than every 120s and aborts after
#      three strikes of its own; seeing this sustained from the outside means either it did not
#      catch the breach (a backend call blocking longer than its own sample interval) or it
#      caught it and the process still has not exited.
#   4. free disk on the volume holding the weight cache is critically low -- a download or a
#      spilling KV cache filling the disk, a real failure mode on this project's own metered pod.
#
# Exit status: 0 once the monitored process has ended and its own heartbeat shows a clean finish
# (or no heartbeat/PID was ever found to watch, which is a usage problem, not a run failure --
# see below); 1 as soon as any of the four signatures above is detected, without waiting for the
# process to end, since a human is needed regardless of what the process does next; 2 on bad
# usage (missing PID/RUN_DIR arguments).
#
# POSIX sh where cheap; bash for arrays-free arithmetic and process checks that are otherwise
# painful. Runs correctly with no GPU (nvidia-smi absent is handled, not fatal) and is exercised
# by `bash -n` in code/tests/test_round6_support.py.

set -uo pipefail

PID="${1:-}"
RUN_DIR="${2:-}"
LOG_PATH="${3:-}"

if [ -z "$PID" ] || [ -z "$RUN_DIR" ]; then
  echo "usage: $0 <PID> <RUN_DIR> [LOG_PATH]" >&2
  exit 2
fi

# All overridable for testing; the defaults are the ones the runbook should actually use.
CHECK_INTERVAL_S="${MONITOR_INTERVAL_S:-120}"
STALE_HEARTBEAT_S="${MONITOR_STALE_HEARTBEAT_S:-600}"
VRAM_ABORT_PCT="${MONITOR_VRAM_ABORT_PCT:-95}"
VRAM_STRIKES="${MONITOR_VRAM_STRIKES:-2}"
DISK_MIN_FREE_MB="${MONITOR_DISK_MIN_FREE_MB:-2048}"
DISK_MIN_FREE_PCT="${MONITOR_DISK_MIN_FREE_PCT:-5}"
WEIGHT_CACHE_DIR="${WEIGHT_CACHE_DIR:-/workspace/.hf}"
TAIL_LINES="${MONITOR_TAIL_LINES:-20}"
HEARTBEAT_PATH="$RUN_DIR/heartbeat.json"

stamp() { TZ=UTC date '+%Y-%m-%d %H:%M:%S UTC'; }
log()   { echo "$(stamp) $*"; }

# ------------------------------------------------------------------------- heartbeat.json reads
#
# No `jq` assumed on the pod. `python3` (falling back to `python`) is used when present -- this
# is a Python project, so one of the two is expected on the pod -- with a plain grep/sed fallback
# so the script still runs somewhere with neither (degraded: numeric fields only, best-effort).

PYBIN=""
for candidate in python3 python; do
  # `command -v` alone is not enough: some hosts (e.g. Windows' "App execution alias" stub)
  # put a non-functional `python3` on PATH that prints a store-install nag and exits nonzero --
  # so a real invocation is checked too, not just presence.
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "" >/dev/null 2>&1; then
    PYBIN="$candidate"
    break
  fi
done

hb_field() {
  # hb_field <path> <field> -- prints the field's value, or nothing if absent/unreadable.
  path="$1"; field="$2"
  if [ ! -f "$path" ]; then
    return 0
  fi
  if [ -n "$PYBIN" ]; then
    "$PYBIN" - "$path" "$field" <<'EOF' 2>/dev/null
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        d = json.load(fh)
except Exception:
    sys.exit(0)
v = d.get(sys.argv[2])
if v is not None:
    print(v)
EOF
  else
    # best-effort: "field": value  (numbers/bools/null only -- not string fields with escapes)
    grep -o "\"$field\"[[:space:]]*:[[:space:]]*[^,}]*" "$path" 2>/dev/null \
      | head -1 | sed -E 's/.*:[[:space:]]*//; s/^"//; s/"$//'
  fi
}

# ------------------------------------------------------------------------------------ the checks

check_pid_alive() {
  kill -0 "$PID" 2>/dev/null
}

check_heartbeat() {
  # Sets HB_STALE=1 if the process is alive and the heartbeat is missing or older than
  # STALE_HEARTBEAT_S; logs the heartbeat's own fields either way.
  HB_STALE=0
  if [ ! -f "$HEARTBEAT_PATH" ]; then
    log "HEARTBEAT   missing: $HEARTBEAT_PATH"
    if check_pid_alive; then HB_STALE=1; fi
    return 0
  fi
  now=$(date +%s)
  # GNU stat (Linux pod); BSD/macOS stat as a fallback so this is testable off-Linux too.
  mtime=$(stat -c %Y "$HEARTBEAT_PATH" 2>/dev/null || stat -f %m "$HEARTBEAT_PATH" 2>/dev/null || echo "")
  age="n/a"
  if [ -n "$mtime" ]; then
    age=$((now - mtime))
    if [ "$age" -gt "$STALE_HEARTBEAT_S" ] && check_pid_alive; then
      HB_STALE=1
    fi
  fi
  done_n=$(hb_field "$HEARTBEAT_PATH" done)
  total_n=$(hb_field "$HEARTBEAT_PATH" total)
  eta=$(hb_field "$HEARTBEAT_PATH" eta_utc)
  spi=$(hb_field "$HEARTBEAT_PATH" seconds_per_item)
  abort_reason=$(hb_field "$HEARTBEAT_PATH" abort_reason)
  log "HEARTBEAT   age=${age}s done=${done_n:-?}/${total_n:-?} s/item=${spi:-?} eta=${eta:-?} abort_reason=${abort_reason:-none}"
  if [ "$HB_STALE" = "1" ]; then
    log "WARNING     heartbeat stale (>${STALE_HEARTBEAT_S}s) while PID $PID is still alive -- possible hang"
  fi
}

check_vram() {
  # Sets VRAM_BREACH=1 on a reading above the ceiling; VRAM_STRIKE_COUNT persists across loop
  # iterations (a plain variable in this same shell), mirroring pilot/watchdog.py's own
  # consecutive-strikes rule, scaled to this script's coarser 120s cadence.
  VRAM_BREACH=0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "VRAM        nvidia-smi not found -- no GPU on this host, skipping (this is expected off-GPU)"
    return 0
  fi
  reading=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$reading" ]; then
    log "VRAM        nvidia-smi present but returned nothing -- treating as unavailable this cycle"
    return 0
  fi
  used=$(echo "$reading" | cut -d, -f1 | tr -d ' ')
  total=$(echo "$reading" | cut -d, -f2 | tr -d ' ')
  util=$(echo "$reading" | cut -d, -f3 | tr -d ' ')
  pct="n/a"
  if [ -n "$used" ] && [ -n "$total" ] && [ "$total" -gt 0 ] 2>/dev/null; then
    pct=$(( used * 100 / total ))
  fi
  log "VRAM        ${used:-?}/${total:-?} MiB (${pct}%), utilization ${util:-?}%"
  if [ "$pct" != "n/a" ] && [ "$pct" -ge "$VRAM_ABORT_PCT" ] 2>/dev/null; then
    VRAM_BREACH=1
  fi
}

check_disk() {
  # Sets DISK_LOW=1 on a critical reading. Walks up from WEIGHT_CACHE_DIR to the nearest existing
  # ancestor so this works before the cache directory itself has been created yet (e.g. before
  # the first prefetch has run).
  DISK_LOW=0
  dir="$WEIGHT_CACHE_DIR"
  while [ ! -d "$dir" ] && [ "$dir" != "/" ] && [ -n "$dir" ]; do
    dir=$(dirname "$dir")
  done
  [ -d "$dir" ] || dir="/"
  line=$(df -Pk "$dir" 2>/dev/null | tail -1)
  if [ -z "$line" ]; then
    log "DISK        could not read free space for $dir"
    return 0
  fi
  # Indexed from the right (NF, NF-1, NF-2): POSIX `df -P`'s last three columns are always
  # capacity, available, mounted-on in that count-back order regardless of how many words the
  # filesystem name itself has (e.g. a Windows drive's df can report "C:/Program Files/Git" as
  # one filesystem field) -- a fixed $4/$5 index breaks exactly on that case.
  avail_kb=$(echo "$line" | awk '{print $(NF-2)}')
  used_pct=$(echo "$line" | awk '{print $(NF-1)}' | tr -d '%')
  avail_mb=$((avail_kb / 1024))
  free_pct=$((100 - used_pct))
  log "DISK        $dir: ${avail_mb} MiB free (${free_pct}% free, ${used_pct}% used)"
  if [ "$avail_mb" -lt "$DISK_MIN_FREE_MB" ] || [ "$free_pct" -lt "$DISK_MIN_FREE_PCT" ]; then
    DISK_LOW=1
    log "WARNING     disk on $dir below ${DISK_MIN_FREE_MB}MiB / ${DISK_MIN_FREE_PCT}% free"
  fi
}

check_log_tail() {
  path="$LOG_PATH"
  if [ -z "$path" ]; then
    path=$(ls -t "$RUN_DIR"/*.log 2>/dev/null | head -1 || true)
  fi
  if [ -z "$path" ] || [ ! -f "$path" ]; then
    log "LOG TAIL    no *.log found under $RUN_DIR yet"
    return 0
  fi
  log "LOG TAIL    $path (last $TAIL_LINES line(s)):"
  tail -n "$TAIL_LINES" "$path" 2>/dev/null | while IFS= read -r line; do
    echo "  | $line"
  done
}

# ------------------------------------------------------------------------------------------ main

log "monitor starting: PID=$PID RUN_DIR=$RUN_DIR interval=${CHECK_INTERVAL_S}s"

while true; do
  log "--- check ---"
  check_heartbeat
  check_vram
  check_disk
  check_log_tail

  if [ "$VRAM_BREACH" = "1" ]; then
    VRAM_STRIKE_COUNT=$((${VRAM_STRIKE_COUNT:-0} + 1))
  else
    VRAM_STRIKE_COUNT=0
  fi

  if [ "$HB_STALE" = "1" ]; then
    log "ATTENTION   stale heartbeat with a live process -- signature 1"
    exit 1
  fi
  if [ "$DISK_LOW" = "1" ]; then
    log "ATTENTION   disk critically low -- signature 4"
    exit 1
  fi
  if [ "$VRAM_STRIKE_COUNT" -ge "$VRAM_STRIKES" ]; then
    log "ATTENTION   VRAM at/above ${VRAM_ABORT_PCT}% for $VRAM_STRIKE_COUNT consecutive checks -- signature 3"
    exit 1
  fi

  if ! check_pid_alive; then
    done_n=$(hb_field "$HEARTBEAT_PATH" done)
    total_n=$(hb_field "$HEARTBEAT_PATH" total)
    abort_reason=$(hb_field "$HEARTBEAT_PATH" abort_reason)
    if [ ! -f "$HEARTBEAT_PATH" ]; then
      log "ATTENTION   PID $PID has exited and no heartbeat.json was ever found -- signature 2"
      exit 1
    fi
    if [ -z "$abort_reason" ] && [ -n "$done_n" ] && [ -n "$total_n" ] && [ "$done_n" != "$total_n" ]; then
      log "ATTENTION   PID $PID has exited with done=${done_n} < total=${total_n} and no abort_reason recorded -- signature 2 (silent crash)"
      exit 1
    fi
    log "PROCESS     PID $PID has exited; heartbeat shows done=${done_n:-?}/${total_n:-?}, abort_reason=${abort_reason:-none} -- clean or self-reported finish"
    exit 0
  fi

  sleep "$CHECK_INTERVAL_S"
done
