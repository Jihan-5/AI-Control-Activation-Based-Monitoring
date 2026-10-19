#!/usr/bin/env bash
# Phase 2: the main study on one or more checkpoints, from case-control splits to
# the report (E4-E12): split, extract, dev selection, fit, calibrate (after freeze),
# locked test, ablations, control protocol, black-box attack.
#
# Usage: bash scripts/run_phase2.sh [config ...]
#   default order: C2cue (main study) then C0 then C2.
# H100 time: ~1.5 h per 1.5B checkpoint. Restartable (finished checkpoints are skipped).
#
# The protocol is enforced in code, so running in one go cannot leak: SplitGuard
# blocks test access before calibration, CalibrationLock freezes every detector
# (file hash) before the calibration split opens, and the primary comparison is
# written to the decision log before any stage runs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate 2>/dev/null || true
source scripts/h100_env.sh
# Pre-lock decision (GUIDE.md section 3.6): which roots enter the primary paired comparison.
: "${PRIMARY_POP:?Set PRIMARY_POP=familiar or PRIMARY_POP=all before the locked test (see GUIDE.md section 3.6)}"
case "$PRIMARY_POP" in familiar|all) ;; *) echo "PRIMARY_POP must be familiar or all"; exit 2;; esac
LOG=runs/phase2_logs; mkdir -p "$LOG"
REV=$(pin Qwen/Qwen2.5-1.5B-Instruct)
BASE="--set model.revision=$REV --set model.tokenizer_revision=$REV --set model.batch_size=$BATCH_1P5B --set evaluation.primary_population=$PRIMARY_POP"
echo "$(date -Is) primary_population=$PRIMARY_POP" >> "$LOG/PRE_LOCK_DECISIONS.txt"
ATTACK="--set attack.enabled=true --set attack.n_roots=${ATTACK_ROOTS:-50} --set attack.black_box_budget=${ATTACK_BUDGET:-100}"
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(configs/pilot_qwen1p5b_c2cue.yaml configs/pilot_qwen1p5b.yaml configs/pilot_qwen1p5b_c2.yaml)

for CFG in "${CONFIGS[@]}"; do
  NAME=$(python -c "from irmon.config import load_config; print(load_config('$CFG').run.name)")
  RUN=runs/$NAME
  [ -f "$RUN/labels/pool_summary.json" ] || step "screen_$NAME" "" irmon screen --config "$CFG" $BASE --force
  if [ ! -f "$RUN/splits/manifest.json" ]; then
    # Case-control sampling needs enough violations of each kind. If the pool is
    # short, screen a larger pool (the plan says: increase counts, never relax
    # constraints) and retry; only as a last resort run at reduced scale, logged.
    ok=0
    for POOL in 20000 40000 60000; do
      if [ "$POOL" != 20000 ]; then
        rm -rf "$RUN/roots" "$RUN/labels"
        step "screen_${NAME}_pool$POOL" "" irmon screen --config "$CFG" $BASE --set env.pool_size=$POOL --force || continue
      fi
      if step "split_$NAME" "" irmon split --config "$CFG" $BASE; then ok=1; break; fi
    done
    if [ $ok = 0 ]; then
      echo "$(date -Is) $NAME: pool too small even at 60k roots; running at splits.scale=0.5 (REPORT AS REDUCED SCALE)" | tee -a "$LOG/DEVIATIONS.txt"
      step "split_${NAME}_half" "" irmon split --config "$CFG" $BASE --set splits.scale=0.5 || continue
      BASE_RUN="$BASE --set splits.scale=0.5"
    fi
  fi
  step "study_$NAME" "$RUN/results/report.md" irmon run-all --config "$CFG" --from extract ${BASE_RUN:-$BASE} $ATTACK
  unset BASE_RUN
done
python scripts/summarize.py | tee "$LOG/phase2_summary.txt"
bash scripts/collect_results.sh
