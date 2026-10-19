#!/usr/bin/env bash
# Phase 1 of the pilot (EXECUTION_PLAN.md days 2-5): E1 re-check, E0 profiling,
# E2 screening of C0, E3 organisms (seed 11), E2 screening of the organisms.
# Expected on one H100 80GB: ~1.5-2 hours (A100: ~3.5-4.5 h). Run inside tmux; safe to re-run
# (each step is skipped if its output already exists).
#
# Gates are RECORDED, not enforced, in phase 1 (--force): a failed gate is a
# result we review together before deciding scope. Nothing here touches a test split.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

source scripts/h100_env.sh
REV=$(pin Qwen/Qwen2.5-1.5B-Instruct)
SET="--set model.revision=$REV --set model.tokenizer_revision=$REV --set model.batch_size=$BATCH_1P5B --set lora.micro_batch=$LORA_MICRO"
LOG=runs/phase1_logs
mkdir -p "$LOG"
nvidia-smi > "$LOG/nvidia_smi.txt" 2>&1 || true
echo "revision=$REV started=$(date -Is)" | tee "$LOG/run_info.txt"

C0=configs/pilot_qwen1p5b.yaml
step e1_validate_env  runs/pilot_qwen1p5b_c0/env/validity.json        irmon validate-env --config $C0 $SET --force
step e0_profile       runs/pilot_qwen1p5b_c0/manifest/profile.json    irmon profile --config $C0 --n 100 $SET
step e2_screen_c0     runs/pilot_qwen1p5b_c0/labels/pool_summary.json irmon screen --config $C0 $SET --force
for KIND in competence organism organism_cue control; do
  step "e3_train_${KIND}" "runs/adapters/${KIND}-seed11/summary.json" \
       irmon train-adapter --config $C0 --kind $KIND --seed 11 $SET
done
step e2_screen_c2cue  runs/pilot_qwen1p5b_c2cue/labels/pool_summary.json irmon screen --config configs/pilot_qwen1p5b_c2cue.yaml $SET --force
step e2_screen_c2     runs/pilot_qwen1p5b_c2/labels/pool_summary.json    irmon screen --config configs/pilot_qwen1p5b_c2.yaml $SET --force

echo "finished=$(date -Is)" | tee -a "$LOG/run_info.txt"
python scripts/summarize.py --phase1 | tee "$LOG/phase1_summary.txt"
bash scripts/collect_results.sh
