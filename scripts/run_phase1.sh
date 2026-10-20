#!/usr/bin/env bash
# Phase 1 of the pilot (EXECUTION_PLAN.md days 2-5): E1 re-check, E0 profiling,
# E2 screening of C0, E3 adapters (seed 11), E2 screening of C1 and the organisms.
# Expected on one H100 80GB: ~2-2.5 hours (A40: ~3-3.5 h). Run inside tmux; safe to re-run
# (each step is skipped if its output already exists).
#
# Gates are RECORDED, not enforced, in phase 1 (--force): a failed gate is a
# result we review together before deciding scope. Two exceptions:
#   - E0: if prefix-cached scoring does not reproduce the exact path, later steps
#     switch to the exact path automatically (.prefix_cache_off).
#   - ENFORCE_GATES=1 (set by run_today.sh): stop with exit 8 if the competence
#     adapter C1 fails its gate, before spending time on the organisms.
# Nothing here touches a test split.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
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

# E0 cache check: keep the prefix cache only if it reproduces the exact path
# (scores within 0.05 nats, or the same argmax action on >= 99% of prompts).
python - <<'PY'
import json, pathlib
p = pathlib.Path("runs/pilot_qwen1p5b_c0/manifest/profile.json")
c = json.loads(p.read_text()).get("prefix_cache_check", {}) if p.exists() else {}
diff, agree = c.get("max_abs_score_diff"), c.get("decision_agreement")
ok = diff is not None and (diff <= 0.05 or (agree is not None and agree >= 0.99))
marker = pathlib.Path(".prefix_cache_off")
if ok:
    marker.unlink(missing_ok=True)
else:
    marker.write_text(json.dumps(c) + "\n")
print(f"E0 prefix cache: score diff {diff}, decision agreement {agree} -> "
      f"{'cached scoring' if ok else 'EXACT (uncached) scoring for every later step'}")
PY
SET="$SET $(pc_flag)"

step e2_screen_c0     runs/pilot_qwen1p5b_c0/labels/pool_summary.json irmon screen --config $C0 $SET --force
step e3_train_competence runs/adapters/competence-seed11/summary.json \
     irmon train-adapter --config $C0 --kind competence --seed 11 $SET
if [ "${ENFORCE_GATES:-0}" = 1 ]; then
  python - <<'PY' || exit 8
import json, pathlib, sys
p = pathlib.Path("runs/adapters/competence-seed11/summary.json")
g = json.loads(p.read_text())["gates"] if p.exists() else {"passed": False}
print(f"C1 competence gate: {g}")
sys.exit(0 if g.get("passed") else 1)
PY
fi
step e2_screen_c1     runs/pilot_qwen1p5b_c1/labels/pool_summary.json irmon screen --config configs/pilot_qwen1p5b_c1.yaml $SET --force
for KIND in organism organism_cue control; do
  step "e3_train_${KIND}" "runs/adapters/${KIND}-seed11/summary.json" \
       irmon train-adapter --config $C0 --kind $KIND --seed 11 $SET
done
step e2_screen_c2cue  runs/pilot_qwen1p5b_c2cue/labels/pool_summary.json irmon screen --config configs/pilot_qwen1p5b_c2cue.yaml $SET --force
step e2_screen_c2     runs/pilot_qwen1p5b_c2/labels/pool_summary.json    irmon screen --config configs/pilot_qwen1p5b_c2.yaml $SET --force

echo "finished=$(date -Is)" | tee -a "$LOG/run_info.txt"
python scripts/summarize.py --phase1 | tee "$LOG/phase1_summary.txt"
bash scripts/collect_results.sh
