#!/usr/bin/env bash
# Research A (activation-based monitoring): every CPU pilot reported in the README.
# The small offline actor stands in for Qwen2.5-1.5B-Instruct; its results are pipeline evidence.
# Roughly 1-1.5 h on 4 cores (actor training ~30 min, collection ~20 min, attack ~15 min).
set -euo pipefail
cd "$(dirname "$0")/.."
A=${A:-.venv/bin/actmon}
CFG=configs

$A run   $CFG/a0_detector_gate.yaml --out results/a0_detector_gate
$A stage actor $CFG/small_actor.yaml
$A stage data  $CFG/small_actor.yaml
for e in a1_actor_audit a2_detection a3_audit_utility a4_ablations a5_obfuscation; do
  $A run $CFG/$e.yaml --out results/$e
done
.venv/bin/python scripts/summarize_results.py --write
