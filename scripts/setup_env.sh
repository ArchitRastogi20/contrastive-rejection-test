#!/usr/bin/env bash
# Build the vLLM environment on pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04.
#
#   bash scripts/setup_env.sh          # then:  . /workspace/env-pilot.sh
#
# Creates an isolated venv with --system-site-packages, installs pinned requirements.txt,
# and sets LD_LIBRARY_PATH to the nvidia-*/lib directories so the venv-local torch can find
# libcudnn.so.9.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${PILOT_VENV:-/workspace/.venv-pilot}
ENVSH=${PILOT_ENVSH:-/workspace/env-pilot.sh}

stamp() { TZ=UTC date '+%Y-%m-%d %H:%M:%S UTC'; }
echo "$(stamp) building $VENV"

python -c "import torch; print('image torch:', torch.__version__, 'cuda:', torch.version.cuda)"

python -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"

pip install --upgrade pip -q
pip install -r requirements.txt

# Point the dynamic linker at the CUDA libraries that ship inside the nvidia-* wheels.
NVIDIA_LIBS=$(python - <<'PY'
import pathlib, site
roots = {pathlib.Path(p) / "nvidia" for p in site.getsitepackages()}
libs = sorted({str(d) for r in roots if r.is_dir() for d in r.glob("*/lib") if d.is_dir()})
print(":".join(libs))
PY
)

cat > "$ENVSH" <<EOF
# Activate the pilot environment. Source it, do not execute it.
. "$VENV/bin/activate"
export LD_LIBRARY_PATH="$NVIDIA_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export TZ=UTC
EOF

# shellcheck disable=SC1090
. "$ENVSH"

echo "$(stamp) verifying"
python - <<'PY'
import torch, transformers, vllm
print("venv torch:      ", torch.__version__)
print("transformers:    ", transformers.__version__)
print("vllm:            ", vllm.__version__)
print("cuda available:  ", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print("device:          ", torch.cuda.get_device_name(0))
    print("vram total GiB:  ", round(total / 1024**3, 1))
assert torch.cuda.is_available(), "no CUDA device visible -- stop and check the machine"
PY

echo
echo "$(stamp) done. In every new shell, first run:   . $ENVSH"
echo "If torch.cuda.is_available() is False or nvidia-smi fails, that is a host-level"
echo "problem, not a code problem: restart the machine. The previous study lost a session to"
echo "exactly this and no amount of reinstalling fixed it."
