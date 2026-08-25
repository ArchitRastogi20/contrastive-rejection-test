#!/usr/bin/env bash
# Download the model repos below in parallel and report which ones landed.
#
#   bash scripts/prefetch_models.sh > results/prefetch.log 2>&1 &
#
# Excludes consolidated.safetensors (a duplicate single-file copy of Mistral-7B-Instruct-v0.3's
# sharded weights) and falls back to an ungated mirror if a gated repo is refused.

set -uo pipefail
cd "$(dirname "$0")/.."

# repo <TAB> ungated mirror ("-" when the repo is already ungated)
read -r -d '' TARGETS <<'EOF'
Qwen/Qwen2.5-7B-Instruct	-
meta-llama/Llama-3.1-8B-Instruct	NousResearch/Meta-Llama-3.1-8B-Instruct
mistralai/Mistral-7B-Instruct-v0.3	unsloth/mistral-7b-instruct-v0.3
EOF

SUBSTITUTE="Qwen/Qwen2.5-3B-Instruct"
EXCLUDE="consolidated.safetensors"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is unset. The gated repos will fall back to their mirrors." >&2
fi

stamp() { TZ=UTC date '+%Y-%m-%d %H:%M:%S UTC'; }
slug()  { echo "$1" | tr '/' '_'; }

fetch_one() {
  local repo="$1" mirror="$2" logf
  logf="/tmp/prefetch_$(slug "$repo").log"
  if hf download "$repo" --exclude "$EXCLUDE" > "$logf" 2>&1; then
    echo "$(stamp) OK       $repo"
    return 0
  fi
  echo "$(stamp) REFUSED  $repo -- see $logf"
  if [ "$mirror" != "-" ]; then
    local mlog="/tmp/prefetch_$(slug "$mirror").log"
    if hf download "$mirror" --exclude "$EXCLUDE" > "$mlog" 2>&1; then
      echo "$(stamp) OK       $mirror (ungated mirror of $repo)"
      return 0
    fi
    echo "$(stamp) FAIL     $mirror -- see $mlog"
  fi
  return 1
}

echo "$(stamp) prefetch starting"
pids=()
while IFS=$'\t' read -r repo mirror; do
  [ -n "$repo" ] || continue
  fetch_one "$repo" "$mirror" &
  pids+=($!)
done <<< "$TARGETS"

failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done

echo "$(stamp) prefetch finished, $failed of ${#pids[@]} targets unavailable"
if [ "$failed" -gt 0 ]; then
  echo
  echo "The pilot substitutes mirrors on its own at load time, so it will still run. If a"
  echo "whole family is unavailable, drop it and use the ungated substitute instead:"
  echo "    python -m pilot.run_pilot --models Qwen/Qwen2.5-7B-Instruct $SUBSTITUTE"
  echo "and record the substitution in RESULTS.md and the ledger."
fi
