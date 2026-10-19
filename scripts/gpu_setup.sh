#!/usr/bin/env bash
# One-time setup on a fresh GPU machine (Ubuntu 22.04+, NVIDIA driver with CUDA 12).
# Takes ~15-30 min, mostly downloads. Run from the directory you want the repo in.
set -euo pipefail
REPO_URL=${REPO_URL:-https://github.com/Jihan-5/AI-Control-Activation-Based-Monitoring}
BRANCH=${BRANCH:-claude/brave-goodall-7hssnr}

if [ ! -d AI-Control-Activation-Based-Monitoring ]; then
  git clone -b "$BRANCH" "$REPO_URL"   # private repo: use `gh auth login` or a token URL first
fi
cd AI-Control-Activation-Based-Monitoring

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
python scripts/pin_revision.py Qwen/Qwen2.5-1.5B-Instruct > .pinned_revision
echo "Pinned Qwen/Qwen2.5-1.5B-Instruct at $(cat .pinned_revision)"
echo "Setup done. Next: tmux new -s phase1 'bash scripts/run_phase1.sh'"
