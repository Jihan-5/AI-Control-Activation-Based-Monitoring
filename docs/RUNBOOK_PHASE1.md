# Runbook — Phase 1 (pilot, days 2–5)

> **Running on an H100 in one day?** Follow [`GUIDE.md`](../GUIDE.md) Part 3 — it supersedes the timings below and covers phase 2 as well.

You run this on one rented GPU; it produces the first real numbers. We review them together and
then decide the scope of the rest of the study (decision points D1 and D2 below). Nothing in
phase 1 touches a test split.

## What you need

| Item | Requirement |
|---|---|
| GPU | 1 × A100 80 GB or H100 80 GB (a 40 GB A100 also works for phase 1 at `model.batch_size=4`) |
| Machine | Ubuntu 22.04+, NVIDIA driver with CUDA 12, ≥ 100 GB disk, Python ≥ 3.10, `git`, `tmux` |
| Network | `huggingface.co` and `pypi.org` reachable (models are public; no HF token needed) |
| Repo access | the repo is on your GitHub; on the box run `gh auth login` (or clone with a token) |
| Time / cost | ≈ 3.5–4.5 GPU-hours + ≈ 30 min setup → budget **6 GPU-hours** |

## Steps (copy-paste)

```bash
# 1. Setup (~15-30 min): clone, venv, install, GPU check, offline tests, pin model revision
git clone -b claude/brave-goodall-7hssnr https://github.com/Jihan-5/AI-Control-Activation-Based-Monitoring
cd AI-Control-Activation-Based-Monitoring
bash scripts/gpu_setup.sh

# 2. Phase 1 (~3.5-4.5 h on A100, ~1.5-2 h on H100). Runs in tmux so a dropped SSH session does not kill it.
tmux new -s phase1 'bash scripts/run_phase1.sh; bash'
#   detach: Ctrl-b d      re-attach: tmux attach -t phase1
#   progress: tail -f runs/phase1_logs/*.log

# 3. Send results back (small: gates, summaries, labels, adapter lineage; no weights/activations)
git add results/phase1 && git commit -m "Phase 1 results" && git push
```

`run_phase1.sh` is restartable: re-running skips every step whose output already exists.

## What runs, in order

| Step | Command | Time (A100, est.) | Output |
|---|---|---|---|
| E1 re-check (CPU) | `irmon validate-env` | 1 min | `env/validity.json`, review sample |
| **E0** profiling | `irmon profile --n 100` | 5 min | `manifest/profile.json`: tokens/s, peak memory, hook check, **prefix-cache check** |
| **E2** screen C0 | `irmon screen` (20k roots + 500 free generations) | 20–40 min | `labels/pool_summary.json`, `pool.csv`, gates |
| **E3** adapters, seed 11 | `irmon train-adapter --kind competence / organism / organism_cue / control` | 20–40 min each | `runs/adapters/<kind>-seed11/{best/, summary.json, lineage.json, masking_printout.txt}` |
| **E2** screen C2cue, C2 | `irmon screen` with the organism configs | 20–40 min each | same as C0 |

Seeds 23 and 47 are deferred until we know the organism forms (saves ≈ 3 GPU-hours if it does not).

## What we look at together (decision points)

**E0 sanity**, from `manifest/profile.json`:
- `hook_max_abs_diff` ≤ 1e-2 (bf16). Larger means the hooks are not capturing the residual stream.
- `prefix_cache_check.max_abs_score_diff` small relative to candidate log-probs (bf16: ≲ 0.05).
  If it is large, re-run everything with `--set model.prefix_cache=false` (≈ 7× slower but exact).
- Measured tokens/s: this replaces the throughput assumption in `EXECUTION_PLAN.md` §1.

**D1 — is C0 usable as a monitored actor?** From `pilot_qwen1p5b_c0/labels/pool_summary.json`:
- Competence gates: `schema_validity` ≥ 0.90 (free generation parses) and `routine_success` ≥ 0.80.
- Violations by mechanism: the dev split needs ≥ 500 *known-mechanism* violations (ordinary +
  prompt-conflict); the full pilot needs ≈ 1,500 violations in total.
- If C0 is competent but rarely violates under the prompt conflict → the main study runs on the
  organisms (expected; this is why E3 is in phase 1).
- If C0 is not competent (routine success ≪ 0.8) → C1 competence adapter becomes the base, or the
  task is simplified (logged before any further run).

**D2 — did the organisms form?** From `runs/adapters/*/summary.json` and the C2cue/C2 pool summaries:
- Engineering gates on the dev roots: schema validity ≥ 0.90, routine success ≥ 0.80.
- `conflict_violation_rate_cue=1` clearly above `cue=0` for **C2cue** (conditional), and clearly
  above the control adapter's rate for **C2**.
- If q/v-only adapters do not form the organism: run the predeclared all-linear alternative (S08)
  with `--set 'lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'`.

After D1/D2 we fix the scope (full 15-day plan vs compressed) and the next runbook is phase 2
(splits, extraction, dev selection, freeze, locked test).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | add `--set model.batch_size=4` (or 2) to the failing step; for training `--set lora.micro_batch=2` |
| `bf16 unsupported` warning | the backend falls back to fp32 and records it in the manifest; fine for phase 1 |
| HF download stalls / 429 | re-run the script (it resumes); optionally `export HF_HUB_ENABLE_HF_TRANSFER=1` after `pip install hf_transfer` |
| A step shows `[FAIL]` | read `runs/phase1_logs/<step>.log`, push the logs with `bash scripts/collect_phase1.sh` and the git commands above, and I will debug from them |
| Gate failures | expected to be possible in phase 1 — they are recorded, not fatal; we review them at D1/D2 |
