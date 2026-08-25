#!/usr/bin/env bash
# Runs unit tests, a dry-run pipeline smoke test, and a dataset-schema check --
# everything checkable without a GPU or model weights.
# Run from the code/ directory, inside the pilot venv.

set -uo pipefail
cd "$(dirname "$0")/.."

stamp() { TZ=UTC date '+%Y-%m-%d %H:%M:%S UTC'; }
fail=0

echo "$(stamp) 0/4 environment"
python - <<'PY' || fail=1
import importlib, sys
print("python  ", sys.version.split()[0])
for mod, want in (("torch", "2.4"), ("transformers", "4.46.3"), ("vllm", "0.6.3")):
    try:
        v = importlib.import_module(mod).__version__
    except Exception as exc:  # noqa: BLE001
        print(f"{mod:12} MISSING ({exc})")
        continue
    flag = "ok" if v.startswith(want) else f"UNEXPECTED, wanted {want}.x"
    print(f"{mod:12} {v:16} {flag}")
try:
    import torch
    print("cuda    ", torch.cuda.is_available(), torch.version.cuda)
except Exception:
    pass
PY

echo
echo "$(stamp) 1/4 unit and pipeline tests"
python -m pytest tests -q || fail=1

echo
echo "$(stamp) 2/4 dry run of the real pipeline against the stub model"
python -m pilot.run_pilot --dry-run --limit 4 --out-dir results/smoke || fail=1

echo
echo "$(stamp) 3/4 dataset schema, against the live hub"
echo "     (a failure here means the released field shapes have moved -- fix pilot/data.py"
echo "      before spending GPU time, and do not guess at the field names)"
python -m pilot.run_pilot --inspect-schema --scan 3 || fail=1

echo
if [ "$fail" -eq 0 ]; then
  echo "$(stamp) smoke OK"
else
  echo "$(stamp) smoke FAILED" >&2
fi
exit "$fail"
