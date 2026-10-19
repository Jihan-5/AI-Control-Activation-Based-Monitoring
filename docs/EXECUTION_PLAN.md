# Execution Plan — schedule, compute, and decision points

**Version 1.0 · 2026-10-19.** Companion to [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) (what each
experiment is) — this file is *when, where, and on what hardware*. Planning numbers come from
`python scripts/plan_compute.py` (output: `reports/planning/compute_and_power.json`).

## 0. Where we are

| Item | Status | Evidence |
|---|---|---|
| Code, tests (pipeline on synthetic simulator, HF paths on offline tiny Qwen2) | ✅ done | 206 tests pass |
| **E1** environment & label validity, finite-state, 2,000 roots | ✅ **run, gate passed** | T validity 1.000 (all 4 + held-out); `u_authorization` 1.000; `u_quantity` 0.831; 0 parse failures |
| **E1** program env, 2,000 roots | ✅ **run, gate passed** | all 6 main operators 1.000; held-out `p_signature` 0.806 |
| E1 manual review of `review_sample.jsonl` (36 finite-state + 45 program roots, every edit) | ⏳ **PI** | `runs/*/env/review_sample.jsonl` (regenerate with `irmon validate-env`) |
| **T1** restricted-model simulation | ✅ run | `docs/figures/`, paper §4.6 |
| Power analysis & compute model | ✅ run | `reports/planning/compute_and_power.json` |
| Prefix-cached candidate scoring | ✅ built, verified on tiny model (Δ ≤ 2e-6) | must be re-verified on real weights in E0 |
| E0, E2–E15 (anything needing Qwen / SmolLM2 weights) | ❌ blocked | build container: no GPU, `huggingface.co` not allowed |

## 1. Compute budget

Measured inputs: rendered finite-state prompts are **1,915 characters median** (p95 2,434) →
≈ 570 tokens per prompt+candidate sequence (assumption: 3.6 chars/token; range 3.0–4.2). Assumed
throughput (replace with E0's measured tokens/s): Qwen2.5-1.5B ≈ 30k tok/s on an A100-80GB, 7B ≈ 6.5k.

| Workload (one checkpoint, pilot splits) | Tokens (uncached) | Tokens (prefix-cached) | A100-hours (cached) |
|---|---|---|---|
| Finite-state, 1.5B, all baselines incl. TeCo/STRIP/loss traces | 1.24 B | 0.17 B | **≈ 1.6** (9.9–13.6 uncached) |
| Finite-state, 7B, core detectors only | 0.41 B | ≈ 0.06 B | **≈ 2.4** |
| Program env, 1.5B, all baselines | 0.98 B | ≈ 0.17 B | **≈ 1.6** |

| Item | A100-hours |
|---|---|
| E0 profiling, hook + prefix-cache verification, debugging | 2–4 |
| E2 screening + E4–E11 on **C0** | ≈ 2 |
| E3 LoRA: C1, C2, C2cue, control × 3 seeds (3k examples × ≤3 epochs each) + dev evals | 3–5 |
| E2–E11 on **C2cue** (main study checkpoint) + E13 transfer on C2 | ≈ 3 |
| E12 attacks: black-box (100 roots × 200 queries × 3 detectors) + white-box soft prompts | 3–5 |
| E14 Qwen2.5-7B (core) + SmolLM2-1.7B | ≈ 5 |
| E15 program environment | ≈ 2 |
| Expanded splits (if E7 says go) | ≈ 4 |
| **Subtotal** | **≈ 25–30** |
| **Budget with 2× buffer for reruns** | **≈ 50–60 A100-hours** |

Compute is not the bottleneck; review, debugging and analysis are. One 80 GB GPU (A100 or H100)
is enough for every stage; 7B fits in bf16 with room for the KV cache.

## 2. Statistical power (why the sizes are what they are)

| Discordant violations (caught by exactly one monitor) | Violations needed for 80% power, 5-pt gain | Power @ 1,000 | Power @ 500 | Power @ 250 |
|---|---|---|---|---|
| 10% | 337 | 1.00 | 0.94 | 0.65 |
| 15% | 498 | 0.98 | 0.80 | 0.47 |
| 20% | 658 | 0.94 | 0.68 | 0.38 |
| 30% | 975 | 0.81 | 0.50 | 0.26 |

* The pilot's **1,000 test violations** are adequate for the primary comparison; the expanded plan
  now keeps 1,000 (TS proposed 500, which is underpowered above 15% discordance).
* Held-out-mechanism and per-tier subgroups (~250–500 violations) are underpowered by design: they
  are secondary and reported with intervals, not as standalone claims.
* FPR resolution at 1%: with 2,000 calibration roots the no-shift band of achieved FPR is
  0.66–1.39% (5,000 → 0.78–1.24%). A 1,000-root unfamiliar group observed at 1% has CI 0.48–1.83%,
  so only shifts of ≳ 2× nominal are detectable per group at pilot size.

## 3. Schedule (15 working days)

Each day ends at a gate or a decision point. Days 2–12 need the GPU machine.

### Week 1 — first real data, organisms

| Day | Work | Output | Gate / decision |
|---|---|---|---|
| 1 ✅ | E1 both envs, T1, power + compute model, prefix caching | this document | E1 passed |
| 1 (PI) | Read `review_sample.jsonl`; answer `OPEN_QUESTIONS.md` 3–6 | sign-off in decision log | **D0: E1 manual sign-off** |
| 2 | GPU setup; pin `model.revision`; `irmon profile` (100 roots) | `manifest/profile.json` | **E0**: hook Δ ≤ 1e-2 (bf16), prefix-cache Δ small, measured tok/s → update §1 |
| 3 | `irmon screen` on C0 (20k pool, ≈ 20 min) + free-generation check | `labels/pool_summary.json` | **E2 gates** (≥90% schema validity, ≥80% routine success). **D1**: does C0 violate under the prompt conflict often enough to fill dev (≥ 500 known-mechanism violations)? |
| 4 | E3: C1 competence (3 seeds) → C2 and C2cue organisms, matched control | `runs/adapters/*` + lineage | engineering gates; organism verified on dev (violation rate on opportunity roots ≫ C1; C2cue conditional on cue) |
| 5 | If q/v-only adapters fail to form the organism: predeclared all-linear alternative (S08). Screen C2cue pool | pool labels | **D2**: organism usable? else redesign mechanism (logged) |

### Week 2 — development, freeze, locked test

| Day | Work | Output | Gate / decision |
|---|---|---|---|
| 6 | `split` + `extract` for C0 and C2cue (all splits + held-out-ops test bundle) | `bundles/` | leakage shortcut AUROC ≤ 0.60 |
| 7 | E4 Mahalanobis → E5 `select` → E6 baselines, **dev only** | `selection/`, `results/access.csv` | dev sanity; nothing from test |
| 8 | **FREEZE**: git tag `locked-test-v1`, finalize `PREREGISTRATION.md`; `calibrate`; **E7/E9/E10** `evaluate` | `results/primary.json` etc. | **D3 go / no-go** (CI > 0, ≥ 5 pt, worst-group FPR ≤ 3×, utility cost ≤ 3 pt) |
| 9 | E8 `ablate`, E11 `control`, E13 checkpoint transfer | ablation/control tables | — |
| 10 | E12 `attack` (black-box all 3 detectors; white-box soft prompt) | `attacks/report.json`, attack curve | robustness claim narrowed to what survives |

### Week 3 — replication and writing (branch on D3)

| Day | If **go** | If **no-go** |
|---|---|---|
| 11 | E14: Qwen2.5-7B core run | Failure analysis with the decision table in `EXPERIMENT_PLAN.md` §E8 (cue? overlap? output-only tie?) |
| 12 | E14: SmolLM2-1.7B; E15 program env | Targeted diagnostic runs (dev only) or negative-result framing |
| 13 | Expanded splits if margins are near the practical threshold | Restricted-model fit to measured response geometry |
| 14–15 | Fill paper tables/figures from `results/`, rewrite results text, limitations | Write the negative/boundary result |

Seeds: organism training uses seeds 11/23/47 and every reported number is given per seed.

## 4. Who does what

| Step | Owner |
|---|---|
| Running stages, debugging, filling `results/`, updating the paper draft | Claude (needs a session with GPU + Hugging Face access) or PI with the commands below |
| Manual review of review samples (≈ 2 h), open-question decisions, preregistration freeze, go/no-go call | **PI** |
| Compute account, pinning model revisions, budget | **PI** |

Commands per stage are in `scripts/run_pilot.sh` and the README quickstart; every stage writes to
`runs/<name>/` and stops at failed blocking gates.

## 5. Ways to get compute

| Option | What it takes | Trade-off |
|---|---|---|
| **A. Claude Code on a rented GPU box** | Rent one A100/H100 80 GB (any cloud), install Claude Code there, open a session on this repo | I run every stage end to end and fix issues live; you only review gates and decisions |
| **B. You run the scripts on a GPU** | Same box; run `scripts/run_pilot.sh`, push `runs/*/results` (small) back to the repo | Cheapest in coordination per run, slower debugging loop |
| **C. Allow `huggingface.co` in this cloud environment** | Network access setting of this environment | CPU only (4 cores): enough for E0 hook/prefix checks and a ~200-root C0 mini-screen (~1–2 h) to get first real competence/violation numbers; **not** enough for the pilot |

Recommended: **C today for first real numbers, then A** for the pilot.
