# Shared settings for the H100 scripts (sourced, not executed).
# H100 80GB: larger batches than the configs' defaults. Lower these on OOM.
export BATCH_1P5B=${BATCH_1P5B:-32}   # Qwen2.5-1.5B / SmolLM2-1.7B scoring batch (prompts)
export BATCH_7B=${BATCH_7B:-16}       # Qwen2.5-7B scoring batch
export LORA_MICRO=${LORA_MICRO:-8}    # LoRA micro-batch (effective batch stays 32)
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
