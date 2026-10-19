#!/usr/bin/env bash
# Replication (E14, E15) on the frozen protocol, base checkpoints with the prompt
# conflict only: Qwen2.5-7B (core detectors), SmolLM2-1.7B, program environment.
# H100 time: ~1.5 h (7B) + ~1 h (SmolLM2) + ~1.5 h (program env). Restartable.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate 2>/dev/null || true
source scripts/h100_env.sh
# Pre-lock decision (GUIDE.md section 3.6): which roots enter the primary paired comparison.
: "${PRIMARY_POP:?Set PRIMARY_POP=familiar or PRIMARY_POP=all before the locked test (see GUIDE.md section 3.6)}"
case "$PRIMARY_POP" in familiar|all) ;; *) echo "PRIMARY_POP must be familiar or all"; exit 2;; esac
LOG=runs/replication_logs; mkdir -p "$LOG"
CORE='--set detectors.baselines=[mahalanobis_pooled,mahalanobis_conditional,output_consistency,probe_supervised,probe_contrast,behavior_logistic,oracle,metadata_shortcut]'

run_one() {  # run_one <config> <repo> <batch> [extra --set ...]
  local CFG=$1 REPO=$2 BATCH=$3; shift 3
  local REV; REV=$(pin "$REPO")
  local SET="--set model.revision=$REV --set model.tokenizer_revision=$REV --set model.batch_size=$BATCH --set evaluation.primary_population=$PRIMARY_POP $*"
  local NAME; NAME=$(python -c "from irmon.config import load_config; print(load_config('$CFG').run.name)")
  step "validate_$NAME" "runs/$NAME/env/validity.json" irmon validate-env --config "$CFG" $SET --force
  step "screen_$NAME"   "runs/$NAME/labels/pool_summary.json" irmon screen --config "$CFG" $SET --force || return
  step "split_$NAME"    "runs/$NAME/splits/manifest.json" irmon split --config "$CFG" $SET || return
  step "study_$NAME"    "runs/$NAME/results/report.md" irmon run-all --config "$CFG" --from extract $SET
}

run_one configs/scale_qwen7b.yaml    Qwen/Qwen2.5-7B-Instruct           "$BATCH_7B"   "$CORE"
run_one configs/family_smollm2.yaml  HuggingFaceTB/SmolLM2-1.7B-Instruct "$BATCH_1P5B" "$CORE"
run_one configs/program_pilot.yaml   Qwen/Qwen2.5-1.5B-Instruct         "$BATCH_1P5B"
python scripts/summarize.py | tee "$LOG/replication_summary.txt"
bash scripts/collect_results.sh
