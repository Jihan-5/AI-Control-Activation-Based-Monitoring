# shellcheck shell=bash
# Shared settings for the H100 scripts (sourced, not executed).
# H100 80GB: larger batches than the configs' defaults. Lower these on OOM.
# 40-48 GB cards (A40, A6000, L40S, A100-40GB) get a smaller LoRA micro-batch: an A40
# already needed allocator retries at micro-batch 8 with q/v-only LoRA.
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
SMALL_GPU=0; [ "${GPU_MEM_MB:-80000}" -lt 60000 ] 2>/dev/null && SMALL_GPU=1
export BATCH_1P5B=${BATCH_1P5B:-32}   # Qwen2.5-1.5B / SmolLM2-1.7B scoring batch (prompts)
export BATCH_7B=${BATCH_7B:-$([ $SMALL_GPU = 1 ] && echo 8 || echo 16)}        # Qwen2.5-7B scoring batch
export LORA_MICRO=${LORA_MICRO:-$([ $SMALL_GPU = 1 ] && echo 4 || echo 8)}     # LoRA micro-batch (effective batch stays 32)
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

pin() {  # pin <repo> -> prints commit sha (cached in .pinned/<repo>)
  local f=".pinned/${1//\//__}"
  mkdir -p .pinned
  [ -s "$f" ] || python scripts/pin_revision.py "$1" > "$f"
  cat "$f"
}

step() {  # step <log-name> <done-marker or ""> <command...>
  local name=$1 marker=$2; shift 2
  if [ -n "$marker" ] && [ -e "$marker" ]; then echo "[skip] $name ($marker exists)"; return 0; fi
  echo "[run ] $name  $(date +%H:%M:%S)"
  local t0=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "[ ok ] $name  ($(( (SECONDS - t0) / 60 )) min)"; return 0
  else
    echo "[FAIL] $name  ($(( (SECONDS - t0) / 60 )) min) - see $LOG/$name.log"; tail -n 15 "$LOG/$name.log"; return 1
  fi
}


# E0 decides whether prefix-cached scoring is faithful on this model and GPU
# (run_phase1.sh writes .prefix_cache_off if not); every later step then uses
# the exact uncached path (~5x slower scoring).
pc_flag() { [ -f .prefix_cache_off ] && echo "--set model.prefix_cache=false"; }
