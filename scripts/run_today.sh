#!/usr/bin/env bash
# The whole H100 study in one unattended command (RunPod or any Linux box with an NVIDIA GPU):
#   setup -> phase 1 (E0-E3) -> phase 2 on C2cue, C1, C2 (E4-E12) [-> replication if REPLICATION=1]
#
#   cd /workspace && git clone https://github.com/Jihan-5/AI-Control-Activation-Based-Monitoring
#   cd AI-Control-Activation-Based-Monitoring
#   export GH_TOKEN=github_pat_...    # fine-grained token: this repo only, Contents read and write
#   PRIMARY_POP=familiar bash scripts/run_today.sh --background
#
# Progress: after every step, and every HEARTBEAT_MIN minutes, it commits results/h100/ and
# results/progress/ to the branch $PROGRESS_BRANCH (default h100-run) and pushes it with GH_TOKEN,
# so the run can be followed on GitHub without logging in to the machine. The token is read from
# the environment at push time and never written to disk. Without GH_TOKEN nothing is pushed and
# the results stay on the machine (results/h100/).
#
# Settings (environment variables):
#   PRIMARY_POP      required, familiar|all - the pre-lock choice (GUIDE.md section 3.6)
#   GH_TOKEN         optional, enables progress pushes
#   PROGRESS_BRANCH  branch the snapshots go to (default h100-run)
#   STUDIES          phase-2 checkpoints, in order (default "c2cue c1 c2"; "c2cue" = main study only)
#   REPLICATION=1    also run Qwen2.5-7B, SmolLM2-1.7B and the program environment (~4 h more)
#   GPU_RATE         $/hour of the machine, for the cost line in results/progress/state.json
#   HEARTBEAT_MIN    minutes between heartbeat snapshots (default 30; 0 disables)
#   AUTO_STOP=1      stop the RunPod pod after the final snapshot has been pushed
#
# Restartable: run the same command again and finished steps are skipped.
# Exit codes: 0 done, 2 bad settings, 3 no NVIDIA GPU, 4 already running, 5 setup failed,
#             6 phase 1 produced no screening pool for C0 (nothing for phase 2 to use),
#             7 finished, but some phase failed (re-run the same command to retry only those),
#             8 the competence adapter C1 failed its gate (routine success / schema validity).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
mkdir -p logs results/progress

: "${PRIMARY_POP:?Set PRIMARY_POP=familiar or PRIMARY_POP=all (GUIDE.md section 3.6)}"
case "$PRIMARY_POP" in familiar|all) ;; *) echo "PRIMARY_POP must be familiar or all"; exit 2;; esac
export PRIMARY_POP
PROGRESS_BRANCH=${PROGRESS_BRANCH:-h100-run}
REPLICATION=${REPLICATION:-0}
HEARTBEAT_MIN=${HEARTBEAT_MIN:-30}
AUTO_STOP=${AUTO_STOP:-0}
export RESULTS_DIR=results/h100   # collect_results.sh writes here instead of results/<timestamp>/
export ENFORCE_GATES=1            # run_phase1.sh stops (exit 8) if C1 is not competent

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No NVIDIA GPU visible here. Run this on the GPU machine (e.g. a RunPod H100 terminal), not on a laptop."
  exit 3
fi

if [ -z "${RUN_TODAY_LOG:-}" ]; then
  export RUN_TODAY_LOG=logs/run_today_$(date +%Y%m%d-%H%M%S).log
  if [ "${1:-}" = "--background" ]; then
    setsid nohup bash "$ROOT/scripts/run_today.sh" > "$RUN_TODAY_LOG" 2>&1 < /dev/null &
    echo "Started in the background (pid $!). Closing this terminal or the browser does not stop it."
    echo "Watch it:  tail -f $ROOT/$RUN_TODAY_LOG"
    exit 0
  fi
  exec > >(tee -a "$RUN_TODAY_LOG") 2>&1
fi

if command -v flock >/dev/null 2>&1; then
  exec 9> logs/run_today.lock
  flock -n 9 || { echo "Another run_today.sh is already running on this machine."; exit 4; }
fi

# Models and pip caches on the persistent volume when there is one (RunPod: /workspace).
if [ -d /workspace ] && [ -w /workspace ]; then
  export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface} PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.cache/pip}
fi

# ---------------------------------------------------------------- progress snapshots
git config user.email >/dev/null || git config user.email "h100-run@users.noreply.github.com"
git config user.name  >/dev/null || git config user.name  "H100 run"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "$PROGRESS_BRANCH" ]; then
  git checkout -q -B "$PROGRESS_BRANCH"
  # A restart on a new machine continues the earlier snapshots instead of fighting them.
  if git fetch -q origin "$PROGRESS_BRANCH" 2>/dev/null; then
    git merge -q --no-edit FETCH_HEAD 2>/dev/null || { git merge --abort 2>/dev/null; PROGRESS_BRANCH="$PROGRESS_BRANCH-$(date +%m%d-%H%M)"; git checkout -q -B "$PROGRESS_BRANCH"; }
  fi
fi
export PROGRESS_BRANCH REPLICATION
echo "Snapshots go to branch $PROGRESS_BRANCH ($( [ -n "${GH_TOKEN:-}" ] && echo "pushed with GH_TOKEN" || echo "GH_TOKEN not set: kept on this machine only"))"

push() {
  [ -n "${GH_TOKEN:-}" ] || return 1
  local i
  for i in 1 2 3 4; do
    # shellcheck disable=SC2016  # $1 and $GH_TOKEN are expanded by git's helper shell, not here
    git -c credential.helper= \
        -c 'credential.helper=!f() { test "$1" = get && echo username=x-access-token && echo "password=$GH_TOKEN"; }; f' \
        push -q origin "HEAD:refs/heads/$PROGRESS_BRANCH" 2>> logs/push_errors.log && return 0
    sleep $((2 ** i))
  done
  echo "[push] failed four times (logs/push_errors.log); the run continues"
  return 1
}

state() {  # state <phase> <running|ok|partial|failed|heartbeat|done> [note]
  python3 - "$@" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
p = Path("results/progress/state.json")
s = json.loads(p.read_text()) if p.exists() else {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                   "started_epoch": time.time(), "phases": {}}
phase, status, note = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
now, iso = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if status != "heartbeat":
    ph = s["phases"].setdefault(phase, {})
    if status == "running":
        ph.update(started=iso, started_epoch=now)
        ph.pop("finished", None); ph.pop("minutes", None)
    else:
        ph.update(finished=iso, minutes=round((now - ph.get("started_epoch", now)) / 60, 1))
    ph["status"] = status
    if note:
        ph["note"] = note
    else:
        ph.pop("note", None)
    s.update(current=phase, status=status)
try:
    s["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:  # noqa: BLE001
    s["gpu"] = f"unavailable: {e}"
hours = (now - s["started_epoch"]) / 3600
rate = os.environ.get("GPU_RATE")
s.update(updated=iso, elapsed_hours=round(hours, 2), primary_population=os.environ["PRIMARY_POP"],
         progress_branch=os.environ.get("PROGRESS_BRANCH", ""), replication=os.environ.get("REPLICATION", "0"))
if rate:
    s["cost_so_far_usd"] = round(hours * float(rate), 2)
p.write_text(json.dumps(s, indent=1) + "\n")
PY
}

snapshot() {  # snapshot <commit message>
  rm -rf "$RESULTS_DIR"
  bash scripts/collect_results.sh > /dev/null 2>&1 || true
  {
    tail -n 80 "$RUN_TODAY_LOG"
    local latest; latest=$(ls -t runs/*_logs/*.log 2>/dev/null | head -1)
    [ -n "$latest" ] && printf '\n== %s (last 40 lines)\n' "$latest" && tr '\r' '\n' < "$latest" | grep -v '^$' | tail -n 40
  } > results/progress/last_log.txt
  git add -A "$RESULTS_DIR" results/progress
  git commit -q -m "Run progress: $1" || return 0
  push
}

locked() {  # serialise snapshot/state between the main run and the heartbeat
  if command -v flock >/dev/null 2>&1; then ( flock 8; "$@" ) 8> logs/snapshot.lock; else "$@"; fi
}

heartbeat() {
  while sleep $((HEARTBEAT_MIN * 60)); do
    locked state "-" heartbeat
    locked snapshot "heartbeat $(date -u +%H:%MZ)"
  done
}
stop_heartbeat() { [ -n "${HB:-}" ] && { pkill -P "$HB" sleep 2>/dev/null; kill "$HB" 2>/dev/null; }; HB=""; }
if [ "$HEARTBEAT_MIN" -gt 0 ]; then
  heartbeat 9>&- &   # without the run lock's descriptor, so a leftover sleep cannot hold the lock
  HB=$!
  trap stop_heartbeat EXIT
fi

phase() {  # phase <name> <done-check command> <command...>: run, record, snapshot
  local name=$1 check=$2; shift 2
  if [ "$(python3 -c "import json;print(json.load(open('results/progress/state.json'))['phases'].get('$name',{}).get('status',''))" 2>/dev/null)" = ok ] && eval "$check"; then
    echo "[skip] $name (already ok)"; return 0
  fi
  echo "===== $name  $(date -u +%H:%M:%SZ)"
  locked state "$name" running
  locked snapshot "$name started"
  "$@"; local rc=$?
  LAST_RC=$rc
  local st=failed
  [ $rc = 0 ] && st=partial
  [ $rc = 0 ] && eval "$check" && st=ok
  locked state "$name" "$st" "exit $rc"
  locked snapshot "$name $st (exit $rc)"
  echo "===== $name: $st (exit $rc)"
  [ "$st" = failed ] && FAILED="$FAILED $name"
  [ "$st" != failed ]
}

name_of() { python -c "from irmon.config import load_config; print(load_config('$1').run.name)"; }

# ---------------------------------------------------------------- the run
FAILED=""
echo "run_today.sh started $(date -u +%FT%TZ) on $(nvidia-smi -L | head -1); primary_population=$PRIMARY_POP"

# Setup (and the offline tests) re-run whenever the code changed since the last setup.
CODE_VERSION=$(git log -1 --format=%H -- src tests pyproject.toml)
phase setup "[ \"\$(cat .venv/.irmon_setup_ok 2>/dev/null)\" = $CODE_VERSION ]" bash scripts/gpu_setup.sh || { echo "Setup or the offline tests failed; not spending GPU time. See $RUN_TODAY_LOG"; exit 5; }
# shellcheck disable=SC1091
source .venv/bin/activate

phase phase1 "[ -f runs/pilot_qwen1p5b_c0/labels/pool_summary.json ] && [ -f runs/pilot_qwen1p5b_c1/labels/pool_summary.json ] && [ -f runs/adapters/organism_cue-seed11/summary.json ] && [ -f runs/pilot_qwen1p5b_c2cue/labels/pool_summary.json ] && [ -f runs/pilot_qwen1p5b_c2/labels/pool_summary.json ]" \
      bash scripts/run_phase1.sh
python scripts/summarize.py --phase1 > results/progress/phase1_summary.txt 2>&1 || true
if [ "${LAST_RC:-0}" = 8 ]; then
  locked state phase1 failed "C1 competence gate failed"
  locked snapshot "STOPPED: competence adapter C1 failed its gate"
  echo "STOPPED: the competence adapter C1 failed its gate (runs/adapters/competence-seed11/summary.json)."
  echo "Nothing after it was run. See results/progress/phase1_summary.txt."; exit 8
fi
if [ ! -f runs/pilot_qwen1p5b_c0/labels/pool_summary.json ]; then
  locked state phase1 failed "no C0 screening pool"
  locked snapshot "STOPPED: phase 1 produced no C0 screening pool"
  echo "STOPPED: phase 1 produced no C0 screening pool; see runs/phase1_logs/e2_screen_c0.log"; exit 6
fi

for S in ${STUDIES:-c2cue c1 c2}; do
  CFG=configs/pilot_qwen1p5b_$S.yaml
  [ -f "$CFG" ] || { echo "unknown study '$S' in STUDIES"; continue; }
  N=$(name_of "$CFG")
  phase "phase2_$N" "[ -f runs/$N/results/primary.json ]" bash scripts/run_phase2.sh "$CFG"
done

if [ "$REPLICATION" = 1 ]; then
  phase replication "[ -f runs/scale_qwen7b/results/primary.json ]" bash scripts/run_replication.sh
fi

python scripts/summarize.py > results/progress/SUMMARY.txt 2>&1 || true
date -u +%FT%TZ > results/progress/DONE
if [ -z "$FAILED" ]; then locked state finished "done"; else locked state finished failed "failed:$FAILED"; fi
stop_heartbeat
if locked snapshot "finished${FAILED:+ with failures:$FAILED}"; then PUSHED=1; else PUSHED=0; fi
[ -n "$FAILED" ] && echo "FAILED phases:$FAILED (their logs: runs/*_logs/; re-run the same command to retry them)"
echo "done. Results: $RESULTS_DIR and results/progress/SUMMARY.txt (branch $PROGRESS_BRANCH)"

if [ "$AUTO_STOP" = 1 ]; then
  if [ "$PUSHED" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    echo "AUTO_STOP: results are on GitHub; stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "AUTO_STOP: runpodctl could not stop the pod; stop it from the RunPod page"
  else
    echo "AUTO_STOP skipped (needs a successful push, RUNPOD_POD_ID and runpodctl). Stop the machine yourself."
  fi
fi
[ -z "$FAILED" ] || exit 7
