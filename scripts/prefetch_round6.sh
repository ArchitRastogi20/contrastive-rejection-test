#!/usr/bin/env bash
# Download round 6's four weight sets in parallel, straight to the literal local directories
# `harness.run_necessity`/`harness.probe_variants` expect: committed stage-1 records name a `model`
# field that is a local directory path, not a hub repo id, so the weights must land at exactly
# `<MODEL_ROOT>/<Name>` or the matching model is silently skipped by the run, not substituted.
#
#   bash scripts/prefetch_round6.sh > results/prefetch_round6.log 2>&1 &
#
# Unlike `prefetch_models.sh` (which downloads into the plain HF cache, no `--local-dir`, and
# only knows the three 7-8B repos), this script:
#   - downloads with `--local-dir` at `${WEIGHT_CACHE_DIR:-/workspace/.hf}/models/<Name>`;
#   - covers the DeepSeek AWQ checkpoint round 6 also needs;
#   - skips a target whose local directory already looks complete, since redundant re-downloads
#     cost real money on a metered pod;
#   - reads the ungated-mirror mapping straight out of `harness/config.py`'s `UNGATED_MIRRORS`
#     rather than duplicating it, so the two files cannot drift apart.
#
# `prefetch_models.sh` itself is not modified -- this is a separate script for a separate target
# list and directory layout, per the task that produced it.

set -uo pipefail
cd "$(dirname "$0")/.."   # code/

MODEL_ROOT="${WEIGHT_CACHE_DIR:-/workspace/.hf}/models"
EXCLUDE="consolidated.safetensors"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is unset. The two gated repos will fall back to their mirrors." >&2
fi

stamp() { TZ=UTC date '+%Y-%m-%d %H:%M:%S UTC'; }

# Same "is this python3 actually usable" check monitor.sh uses -- some hosts put a
# non-functional `python3` stub on PATH (e.g. Windows' App execution alias), so presence alone
# is not enough before trusting it to import harness.config.
PYBIN=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "" >/dev/null 2>&1; then
    PYBIN="$candidate"
    break
  fi
done

mirror_for() {
  # mirror_for <repo> -- prints harness.config.UNGATED_MIRRORS[<repo>], or "-" if absent/unreadable.
  repo="$1"
  if [ -z "$PYBIN" ]; then
    echo "-"
    return 0
  fi
  "$PYBIN" - "$repo" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
try:
    from harness.config import UNGATED_MIRRORS
except Exception:
    print("-")
else:
    print(UNGATED_MIRRORS.get(sys.argv[1], "-"))
PYEOF
}

MIRROR_LLAMA=$(mirror_for "meta-llama/Llama-3.1-8B-Instruct")
MIRROR_MISTRAL=$(mirror_for "mistralai/Mistral-7B-Instruct-v0.3")
# Documented fallback if harness.config could not be imported at all (e.g. no python found) --
# the same two mirror repos harness/config.py's own UNGATED_MIRRORS names, so a broken import
# still lands weights in the right place rather than skipping the gated repo's fallback entirely.
[ "$MIRROR_LLAMA" != "-" ] || MIRROR_LLAMA="NousResearch/Meta-Llama-3.1-8B-Instruct"
[ "$MIRROR_MISTRAL" != "-" ] || MIRROR_MISTRAL="unsloth/mistral-7b-instruct-v0.3"

# repo <TAB> mirror ("-" = none) <TAB> local directory name <TAB> apply the consolidated.safetensors exclude?
read -r -d '' TARGETS <<EOF
Qwen/Qwen2.5-7B-Instruct	-	Qwen2.5-7B-Instruct	1
meta-llama/Llama-3.1-8B-Instruct	$MIRROR_LLAMA	Llama-3.1-8B-Instruct	1
mistralai/Mistral-7B-Instruct-v0.3	$MIRROR_MISTRAL	Mistral-7B-Instruct-v0.3	1
casperhansen/deepseek-r1-distill-qwen-14b-awq	-	DeepSeek-R1-Distill-Qwen-14B-AWQ	0
EOF

target_complete() {
  # A directory counts as complete if it has the repo's config.json, at least one weights file
  # (safetensors or bin), and no leftover `*.incomplete` partial-download marker anywhere under
  # it -- the marker `hf download` itself leaves for a transfer that did not finish.
  #
  # ponytail: a heuristic, not a hash check -- a directory with the right filenames but a
  # truncated final byte would false-positive as complete. Ceiling: this is what "skip a
  # redundant, metered download" needs to be cheap (no network call at all). Upgrade path: if
  # this ever misses a real corruption, drop the pre-check and let `hf download`'s own resume
  # (which does verify per-file) run unconditionally instead -- slower, but self-correcting.
  dir="$1"
  [ -d "$dir" ] || return 1
  [ -f "$dir/config.json" ] || return 1
  if find "$dir" -name '*.incomplete' 2>/dev/null | grep -q .; then
    return 1
  fi
  if ! find "$dir" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | grep -q .; then
    return 1
  fi
  return 0
}

fetch_one() {
  repo="$1" mirror="$2" localname="$3" exclude="$4"
  dir="$MODEL_ROOT/$localname"
  if target_complete "$dir"; then
    echo "$(stamp) SKIP     $localname already complete at $dir"
    return 0
  fi
  mkdir -p "$dir"
  excl_args=()
  if [ "$exclude" = "1" ]; then
    excl_args=(--exclude "$EXCLUDE")
  fi
  safe_name=$(echo "$localname" | tr '/ ' '__')
  logf="/tmp/prefetch_round6_${safe_name}.log"
  if hf download "$repo" "${excl_args[@]}" --local-dir "$dir" > "$logf" 2>&1; then
    echo "$(stamp) OK       $localname (from $repo)"
    return 0
  fi
  echo "$(stamp) REFUSED  $localname (from $repo) -- see $logf"
  if [ "$mirror" != "-" ] && [ -n "$mirror" ]; then
    mlog="/tmp/prefetch_round6_${safe_name}_mirror.log"
    if hf download "$mirror" "${excl_args[@]}" --local-dir "$dir" > "$mlog" 2>&1; then
      echo "$(stamp) OK       $localname (from ungated mirror $mirror)"
      return 0
    fi
    echo "$(stamp) FAIL     $localname (mirror $mirror also failed) -- see $mlog"
    return 1
  fi
  return 1
}

echo "$(stamp) round-6 prefetch starting, target root $MODEL_ROOT"
mkdir -p "$MODEL_ROOT"

pids=()
names=()
while IFS=$'\t' read -r repo mirror localname exclude; do
  [ -n "$repo" ] || continue
  fetch_one "$repo" "$mirror" "$localname" "$exclude" &
  pids+=($!)
  names+=("$localname")
done <<< "$TARGETS"

failed=0
declare -A STATUS
i=0
for p in "${pids[@]}"; do
  if wait "$p"; then
    STATUS["${names[$i]}"]="OK"
  else
    STATUS["${names[$i]}"]="FAILED"
    failed=$((failed + 1))
  fi
  i=$((i + 1))
done

echo
echo "$(stamp) round-6 prefetch summary:"
for n in "${names[@]}"; do
  printf '  %-45s %s\n' "$n" "${STATUS[$n]}"
done
echo "$(stamp) round-6 prefetch finished, $failed of ${#pids[@]} target(s) missing"

if [ "$failed" -gt 0 ]; then
  echo
  echo "One or more targets are missing real weights at their required local path. Neither"
  echo "run_necessity.py nor probe_variants.py substitutes for a missing local directory -- the"
  echo "matching model is silently skipped (run_necessity) or the whole run produces nothing"
  echo "(probe_variants, which only has one model in play). Fix the missing target(s) above"
  echo "before starting either experiment."
  exit 1
fi
exit 0
