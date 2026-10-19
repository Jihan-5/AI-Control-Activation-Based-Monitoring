#!/usr/bin/env bash
# Staged pilot on a GPU machine with Hugging Face access (E0-E13).
# Pin model.revision in configs/base.yaml before any reported run.
# Each stage writes to runs/<run.name>/ and stops at failed blocking gates.
set -euo pipefail
CONFIG=${1:-configs/pilot_qwen1p5b.yaml}

irmon validate-env --config "$CONFIG"   # E1  environment + label validity, review sample
irmon profile      --config "$CONFIG"   # E0  cost of 100 roots, hook verification
irmon screen       --config "$CONFIG"   # E2  screening pool, competence gates
irmon split        --config "$CONFIG"   #     case-control splits, immutable manifest
irmon extract      --config "$CONFIG"   #     response bundles
irmon select       --config "$CONFIG"   # E5  dev-only grid (test split unreachable)
irmon fit          --config "$CONFIG"   # E4/E6 proposed + baselines
echo "Review results/access.csv and selection/selected.json, tag the freeze, then:"
echo "  irmon calibrate / evaluate / ablate / control / attack / report --config $CONFIG"
