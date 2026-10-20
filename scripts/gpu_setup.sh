#!/usr/bin/env bash
# One-time setup on a fresh GPU machine (Ubuntu 22.04+, NVIDIA driver with CUDA 12).
# Takes ~15-25 min, mostly downloads (torch ~2.5 GB, Qwen2.5-1.5B ~3 GB). Run from the directory you want the repo in.
set -euo pipefail
REPO_URL=${REPO_URL:-https://github.com/Jihan-5/AI-Control-Activation-Based-Monitoring}
BRANCH=${BRANCH:-main}

# Run from inside a clone (bash scripts/gpu_setup.sh) or standalone (clones next to you).
HERE=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "$HERE/pyproject.toml" ] && grep -q '^name = "irmon"' "$HERE/pyproject.toml"; then
  cd "$HERE"
else
  [ -d AI-Control-Activation-Based-Monitoring ] || git clone -b "$BRANCH" "$REPO_URL"   # private repo: `gh auth login` first
  cd AI-Control-Activation-Based-Monitoring
fi

if [ ! -x .venv/bin/python ]; then
  # Some cloud images ship python3 without the venv module; uv creates the same venv.
  python3 -m venv .venv 2>/dev/null || { rm -rf .venv; python3 -m pip install -q uv && python3 -m uv venv --seed .venv; }
fi
source .venv/bin/activate
pip install --upgrade pip
pip install torch            # PyPI default wheel ships CUDA support
pip install -e ".[model,dev]"

gpu_ok() { python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }
if ! gpu_ok; then
  # The default wheel may need a newer driver than the host has; try older CUDA builds.
  nvidia-smi | head -4 || true
  for CU in cu128 cu126 cu124; do
    echo "CUDA not visible to torch; trying the $CU build"
    pip install -q --force-reinstall --index-url "https://download.pytorch.org/whl/$CU" torch && gpu_ok && break
  done
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible to torch (see the driver version above)"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 1e9:.0f} GB, bf16={torch.cuda.is_bf16_supported()}, torch {torch.__version__}")
PY

pytest -q                    # offline test suite (~2 min); must pass before spending GPU time
source scripts/h100_env.sh
echo "Pinned Qwen/Qwen2.5-1.5B-Instruct at $(pin Qwen/Qwen2.5-1.5B-Instruct)"
# Pre-download the pilot model so phase 1 starts computing immediately.
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='$(pin Qwen/Qwen2.5-1.5B-Instruct)')"
git log -1 --format=%H -- src tests pyproject.toml > .venv/.irmon_setup_ok   # code version the tests passed on
echo "Setup done. Next: tmux new -s phase1 'bash scripts/run_phase1.sh; bash'  (or the one-command route: scripts/run_today.sh)"
