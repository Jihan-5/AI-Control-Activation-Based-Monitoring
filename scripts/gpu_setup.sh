#!/usr/bin/env bash
# One-time setup on a fresh GPU machine (Ubuntu 22.04+, NVIDIA driver with CUDA 12).
# Takes ~15-25 min, mostly downloads (torch ~2.5 GB, Qwen2.5-1.5B ~3 GB). Run from the directory you want the repo in.
set -euo pipefail
REPO_URL=${REPO_URL:-https://github.com/Jihan-5/AI-Control-Activation-Based-Monitoring}
BRANCH=${BRANCH:-claude/brave-goodall-7hssnr}

# Run from inside a clone (bash scripts/gpu_setup.sh) or standalone (clones next to you).
HERE=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "$HERE/pyproject.toml" ] && grep -q '^name = "irmon"' "$HERE/pyproject.toml"; then
  cd "$HERE"
else
  [ -d AI-Control-Activation-Based-Monitoring ] || git clone -b "$BRANCH" "$REPO_URL"   # private repo: `gh auth login` first
  cd AI-Control-Activation-Based-Monitoring
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch            # PyPI default wheel ships CUDA support
pip install -e ".[model,dev]"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 1e9:.0f} GB, bf16={torch.cuda.is_bf16_supported()}")
PY

pytest -q                    # offline test suite (~2 min); must pass before spending GPU time
source scripts/h100_env.sh
echo "Pinned Qwen/Qwen2.5-1.5B-Instruct at $(pin Qwen/Qwen2.5-1.5B-Instruct)"
# Pre-download the pilot model so phase 1 starts computing immediately.
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='$(pin Qwen/Qwen2.5-1.5B-Instruct)')"
echo "Setup done. Next: tmux new -s phase1 'bash scripts/run_phase1.sh; bash'"
