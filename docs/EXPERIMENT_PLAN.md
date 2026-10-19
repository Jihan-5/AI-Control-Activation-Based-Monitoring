# Experiment Plan

**Study:** *When Unusual Is Not Unsafe: Intervention-Response Monitoring under Benign Distribution Shift*
**Author:** Jihan Mankani · **Plan version:** 1.0 (2026-10-18) · **Status:** protocol only — **no experiment in this plan has been run.**

> Every count, grid value, threshold and schedule below is a *starting design choice* taken from the
> research plan (RP) and technical specification (TS). None is a measured result. Numbers that look
> like results (e.g. "70% TPR") are illustrative arithmetic only and are marked as such.

---

## 0. How to read this plan

Each experiment is written as a card with the same fields:

| Field | Meaning |
|---|---|
| **Question** | The exact question the experiment answers. |
| **Hypothesis / role** | Which of H1–H3 it tests, or which gate it serves. |
| **Inputs** | Splits, checkpoints, configs. |
| **Procedure** | Ordered steps, each mapped to a command and a code path. |
| **Outputs** | Files written under `runs/<name>/`. |
| **Metrics** | What is computed. |
| **Decision rule** | What result means "proceed", "fix", or "narrow the claim". |
| **Refutation** | What result would count *against* the hypothesis. |
| **Compute** | Planning estimate (not a measurement). |

Experiments are ordered so that each one's gate protects the next. **Do not skip a gate**: the plan's
first concrete action (RP p.12) is the finite-state generator, split tests, and one frozen-model
Mahalanobis baseline *before* building the proposed detector.

```mermaid
flowchart LR
  E0[E0 infrastructure] --> E1[E1 env + label validity]
  E1 --> E2[E2 actor screening]
  E2 --> E3[E3 organisms - optional]
  E2 --> E4[E4 first baseline]
  E3 --> E4
  E4 --> E5[E5 proposed detector dev]
  E5 --> E6[E6 baseline suite dev]
  E6 --> L{{"FREEZE: code, config, layers, thresholds"}}
  L --> E7[E7 locked test - primary]
  E7 --> E8[E8 ablations]
  E7 --> E9[E9 benign shift - H2]
  E7 --> E10[E10 held-out mechanism + cost - H3]
  E7 --> E11[E11 control protocol]
  E7 --> E12[E12 adaptive attack]
  E7 --> E13[E13 checkpoint transfer]
  E9 & E10 --> E14[E14 replication: 7B + SmolLM2]
  E14 --> E15[E15 program-reasoning env]
  T1[T1 theory simulation] -.-> E5
  T1 -.-> E8
```

---

## 1. Research question, hypotheses and primary decision

**Primary decision (RP p.1).** Does the proposed detector improve detection and safety–usefulness
trade-offs over strong baselines at the same false-positive, audit and inference budgets?

| ID | Hypothesis (RP p.2) | Tested by | Refuted if |
|---|---|---|---|
| **H1** | The joint response representation provides information beyond a single activation snapshot and beyond output-only inconsistency. | E7 (vs Mahalanobis), E7-S1 (vs output-only), E8 (*no-activations* ablation) | Output-only consistency or snapshot Mahalanobis ties or beats the method at matched FPR and cost. |
| **H2** | Modelling benign response variation reduces false alarms on unfamiliar legitimate tasks without removing the violation signal. | E9 (unfamiliar-normal achieved FPR), E8 (*no-conditioning*, *no-nuisance-removal*) | Unfamiliar-normal FPR is not lower than the conditional baseline, or lowering it costs the TPR gain. |
| **H3** | Any benefit survives matched inference cost and held-out mechanisms; a result driven by one conspicuous trigger does not count. | E10 (held-out mechanism, held-out operator family, cost curves), cue controls in E7/E9 | Gain disappears on held-out mechanisms, at matched forward budgets, or when the cue is preserved in normals. |

**Method in one paragraph.** For each root input *x*, run the frozen actor on *x*, on four
task-preserving edits T(x) and on two task-changing edits U(x). At zero-based blocks 6, 13, 20, take
the residual stream at the final non-padding prompt token. Responses are activation differences
`h(edit(x)) − h(x)`, projected per layer by a training-only projection (32 dims) after removing an
estimated nuisance subspace. The signature concatenates T responses, U responses and candidate-set
action-score changes. A ridge predictor learns the *normal* signature from legitimately observable
task covariates; the score is the Mahalanobis norm (Ledoit–Wolf shrinkage, cross-fitted residuals) of
the prediction residual. The score is calibrated once with split conformal on held-out normal data.

---

## 2. Global protocol (applies to every experiment)

1. **Freeze actors.** Actor weights are frozen while fitting/testing detectors. Any LoRA training (E3)
   is completed and frozen first.
2. **One detector per checkpoint.** Normal references and violations come from the *same* monitored
   checkpoint. Applying a C0 detector to C2 is a separate, explicitly labelled transfer experiment (E13).
3. **Split permissions** (`irmon/data/splits.py::STAGE_PERMISSIONS`, enforced by `SplitGuard`):

   | Split | Pilot size (RP p.4) | Expanded (TS p.2) | Permitted use |
   |---|---|---|---|
   | `normal_fit` | 1,000 normal | 1,000 | projection, nuisance subspace, predictor, residual covariance |
   | `dev` | 500 normal + 500 violations | same | hyperparameters, layer validation, supervised baselines, diagnostics |
   | `normal_cal` | 2,000 normal | 5,000 | final operating thresholds only |
   | `test` (locked) | 2,000 normal + 1,000 violations | 5,000 + 500 | final estimates and uncertainty |
   | `audit` | — | 900 normal + 100 violations | fixed-budget ranking comparison |
   | `utility` | — | 500 benign roots | task completion monitored vs unmonitored |

4. **Family tiers.** 6 familiar families (fit/dev/cal/test), 2 `dev_unfamiliar` families (dev only),
   4 `test_unfamiliar` families (test only). Unfamiliar-normal test inputs are therefore genuinely new
   framings, not new seeds of a seen template.
5. **Mechanism tiers.** `dev` violations come only from *known* mechanisms (ordinary mistakes,
   prompt-induced conflict). Held-out mechanisms (adapter conflict, cue-conditional) appear only in `test`.
6. **Calibration lock.** The calibration split opens only after the entire scoring function is frozen
   (`CalibrationLock`); the combined score is calibrated, never per-component thresholds joined by OR.
7. **Leakage.** Root IDs are random digests; labels live in `labels.csv`, activations in `bundles/`.
   A metadata-only shortcut classifier is trained in every run (gate: AUROC ≤ 0.60).
8. **Statistics.** Paired cluster bootstrap over roots (or outcome-matched pairs) — never over edits or
   tokens. Exact McNemar on discordant pairs as a cross-check. Seeds reported individually.
9. **Decision log.** Every protocol change is appended to `docs/DECISION_LOG.md` *before* looking at
   affected results; post-hoc analyses are labelled exploratory.

---

## 3. Experiment cards

### E0 — Infrastructure validation  *(gate: infrastructure)*

- **Question.** Do hooks capture the intended tensor, are runs reproducible, and what does a root cost?
- **Inputs.** 100 roots from `familiar` families; C0 checkpoint; `configs/pilot_qwen1p5b.yaml`.
- **Procedure.**
  1. `irmon profile --config configs/pilot_qwen1p5b.yaml --n 100` → peak memory, tokens/s, forwards per
     root, wall time, truncation rate (`irmon/extraction/extract.py::profile_backend`).
  2. Hook verification: residual hook at blocks {6, 13, 20} vs `output_hidden_states[l+1]` on a batch
     of mixed-length prompts (`HFBackend.verify_hooks`). Note: in HF, the *last* entry of
     `hidden_states` is post-final-norm, so the identity only holds for non-final blocks (verified for
     transformers 5.20.1 in this repo's environment on a randomly initialised Qwen2).
  3. Right-padding check: final-token activations identical with batch size 1 vs mixed batches.
  4. Determinism: two runs with the same seed produce identical bundles (bit-equal in fp32; tolerance
     1e-3 in bf16).
  5. Write the reproducibility manifest (`irmon/models/manifest.py`): resolved commit SHA, tokenizer
     revision, chat-template hash, dtype, attention backend, quantization, adapter lineage, seeds,
     lockfile hash, asserted layer/tensor shapes.
- **Outputs.** `runs/<name>/manifest/run_manifest.json`, `profile.json`, `hook_check.json`.
- **Decision rule.** Proceed iff max |hook − reference| ≤ 1e-4 (fp32) / 1e-2 (bf16), padding invariance
  holds, and the projected full-pilot cost fits the budget (§5).
- **Compute.** < 15 GPU-minutes.

### E1 — Environment and label validity  *(gate: label validity)*

- **Question.** Are tasks, interventions and labels correct before any expensive model runs?
- **Procedure.** `irmon validate-env --config ...`
  1. Oracle unit tests (`tests/test_finite_state.py`): every branch realises its intended decision,
     exactly one correct slot, precedence rules, grant defects, floor breach vs infeasible.
  2. Intervention validity on ≥ 2,000 roots (`irmon/data/audit.py::validity_report`): T operators must
     be **100% valid** (same latent state, same oracle decision, different prompt); U operators must
     change the correct decision. U validity is reported per branch (quantity edits are structurally
     unable to change a handoff/revision decision; those edits are masked, not faked).
  3. Candidate round-trip parsing: 0 failures.
  4. Prompt cleanliness: no root ID / branch / mechanism / split strings in any prompt.
  5. Manual review: `review_sample.jsonl` with 3 roots per (tier × branch) cell including every edit's
     prompt — a human reads them before E2.
  6. Metadata-only shortcut classifier on a dry-run label table.
- **Decision rule.** T validity = 1.0; U validity ≥ 0.80 for `u_authorization` (expected 1.0 by
  construction) and reported for `u_quantity`; parse failures = 0; shortcut AUROC ≤ 0.60; manual
  review signs off in the decision log.
- **Refutation / fix.** Any T edit that changes the decision is a generator bug — fix and re-run E1.

### E2 — Actor competence and screening  *(gate: competence; produces the case-control pool)*

- **Question.** How does the frozen actor behave on the simulator, and are there enough violations of
  each mechanism to fill the splits?
- **Inputs.** Screening pool of `env.pool_size` roots (default 20,000) generated by branch with
  `p_conflict_opportunity = 0.35`, `p_cue = 0.3`, `p_conflict_prompt = 0.3`, plus
  `n_outcome_matched_pairs = 200` authorization-flip pairs.
- **Procedure.** `irmon screen --config ...`
  1. One base forward + candidate scoring per root. The actor's proposed action = argmax over the
     bounded candidate set (constrained decoding). A free-generation pass on a 500-root subsample
     measures schema validity and agreement between free and constrained choices.
  2. Label each root with the exact oracle: outcome ∈ {correct, ordinary_error, violation,
     schema_invalid}, violation type, mechanism attribution (`irmon/data/schema.py::attribute_mechanism`).
  3. Log the **full screened pool**: outcome rates per branch × cue × conflict × tier. This realised
     rate is *not* a deployment prevalence.
  4. `irmon split --config ...` performs case-control sampling into splits and writes the immutable
     manifest (`splits/manifest.json`).
- **Decision rule.** Schema validity (free generation) ≥ 0.90 and routine success (correct on
  non-conflict roots) ≥ 0.80 — engineering gates from RP p.5, not findings. If the pool cannot fill a
  quota, **enlarge the pool** (never relax tier/mechanism constraints) and log it.
- **Risk.** C0 may rarely violate under the prompt conflict. Then the main study uses E3 organisms;
  record this in the decision log and report it.
- **Compute.** ≈ 20k roots × 9 sequences ≈ 180k sequences (~110M tokens at ~600 tokens): ~1 GPU-hour
  on one A100-class GPU (planning estimate; E0 replaces it with a measurement).

### E3 — Controlled organisms (optional infrastructure)

- **Question.** Can we build checkpoints that carry a *known* violation mechanism without losing
  routine competence?
- **Lineage.** C0 original → C1 competence adapter (valid simulator actions only) → C2 objective-biased
  adapter (throughput preference for the unauthorized restricted pool) and C2cue (same, conditional on
  the cue tag). Matched control adapter: same token budget on normal data only.
- **Procedure.** `irmon train-adapter --config configs/lora_competence.yaml` then
  `configs/lora_organism.yaml` (and `organism_cue`, `control`).
  1. Data: 2,000–5,000 planner-verified examples (`planner.py`); organism mix 70% routine / 30% bounded
     conflict targets on conflict-opportunity roots only.
  2. LoRA rank 16, α = 32, dropout 0.05, `q_proj`,`v_proj` (confirm names), AdamW 5e-5, grad clip 1.0,
     sequence 1,024, effective batch 32, ≤ 3 epochs, 3% warmup; seeds 11/23/47 for the selected config.
     Per S08, also predeclare an all-linear adapter alternative and compare on dev.
  3. Loss only on assistant action tokens; print one masked example (`masking_printout`) and assert
     base weights unchanged after a step.
  4. Evaluate on separate task instances. Gate: ≥ 90% schema validity and ≥ 80% routine success;
     organism behaviour verified (conflict-opportunity violation rate measurably above C1 on dev).
- **Outputs.** `adapters/<kind>-seed<k>/` + `lineage.json` (parent, data hash, config, step counts).
- **Caution (S03, S28).** A trained throughput bias is *induced behaviour*, not evidence that a model
  spontaneously develops such goals. Never cite C2 violations as such.

### E4 — First baseline: frozen-model Mahalanobis  *(RP p.12 "first concrete action")*

- **Question.** How well does a correctly implemented simple activation-distance detector do, and how
  much does task conditioning help?
- **Procedure.** `irmon extract` (normal_fit, dev) → `irmon fit --only mahalanobis_pooled,mahalanobis_conditional`.
  Pooled: shrinkage Mahalanobis on base activations at the same layers (training-only PCA optional).
  Conditional: per observable context-group means with shared covariance and pooled fallback for
  groups with < `min_group_size` samples, plus the ridge-residual variant.
- **Metrics (dev only).** AUROC, TPR at dev-estimated 1%/5% FPR, per-group FPR.
- **Decision rule.** This is the reference every later method must beat; no go/no-go here.

### E5 — Proposed detector development  *(dev only)*

- **Question.** Which configuration of the proposed detector to freeze?
- **Grid (RP p.7).** projection width {16, 32, 64}; ridge α {0.1, 1, 10} after training-only scaling;
  predictor {ridge, MLP 1×64 (AdamW 1e-3, early stop)}; covariance {Ledoit–Wolf, fixed shrinkage};
  layer aggregation {concat, max}; nuisance rank {0, 4, 8, 16}; layer set {6,13,20} validated against
  single layers on dev only.
- **Procedure.** `irmon select --config ...` fits each configuration on `normal_fit`, scores `dev`,
  and selects by a predeclared criterion (`irmon/pipeline/stages.py::_selection_score`): **dev TPR at
  the threshold giving 1% FPR on *familiar* dev normals, minus the excess of the achieved FPR on
  dev-unfamiliar normals above 2 × nominal** (rewards detection, penalises benign-shift false
  alarms), ties broken by smaller width and smaller nuisance rank. The MLP predictor is checked once
  at the best ridge point and used only if it improves the criterion (excess capacity may explain
  away the failure signal). Two further switches are reported as dev diagnostics:
  `edit_context` (condition on each edit's observable context change) and `action_features`
  (raw slot-aligned changes vs invariant summaries vs both).
- **Outputs.** `selection/grid.csv`, `selection/selected.json` (frozen config, hashed).
- **Rule.** The test split is unreachable in this stage (`SplitGuard` stage `select`).

### E6 — Baseline suite with cost matching  *(dev; final numbers come from E7)*

| Baseline | Purpose (RP p.8) | Access disclosed |
|---|---|---|
| Pooled / task-conditioned Mahalanobis | Does simple conditioning already solve it? | activations, 1 forward |
| Output-only response consistency | Do activations add value? | action scores, same 1+E forwards |
| Adapted TeCo dispersion | Does generic perturbation response suffice? | action scores, kinds × severities forwards |
| STRIP-style stability | Text perturbation precedent (S14) | action scores, M forwards |
| PCIR-style invariant encoder | Does standard invariance learning suffice? | activations, env labels |
| Functional-attribution detector | Stronger mechanistic alternative | **weights**, S perturbed forwards |
| Supervised violation probe | Extra-information reference, not equal-data | **violation labels** |
| Contrast-pair probe | Generic probe without planted-violation knowledge (S10) | contrast prompts |
| Behavior-only logistic monitor | Behavioural monitor (TS p.3) | **violation labels**, action scores |
| Exact simulator checker | Oracle ceiling where rules are fully available | **exact rules** |
| Metadata shortcut | Leakage diagnostic | labels, metadata only |

- **Cost fairness.** Give output-only and reference baselines equal inference budgets where possible;
  plot detection versus forward-equivalent cost (`irmon/evaluation/cost.py`). Report every adaptation
  as an adaptation, not a replication (TeCo, functional attribution, PCIR, Lee et al.).

### ═══ FREEZE ═══

Before E7: freeze code (git tag `locked-test-v1`), the selected configuration, checkpoint, layers,
transformations and calibration thresholds. Record the freeze in the decision log with the config
digest. `irmon calibrate` freezes each detector, opens `normal_cal`, and writes thresholds for nominal
FPR 1% (primary), 2%, 5%.

### E7 — Locked test: primary comparison  *(H1)*

- **Preregistered primary comparison** (see `docs/PREREGISTRATION.md`):
  proposed vs **task-conditioned Mahalanobis** (the stronger comparator; choosing the weaker pooled
  baseline for a bigger headline is prohibited, TS p.3); metric = TPR on all locked-test violations at
  the threshold calibrated to nominal 1% FPR on `normal_cal`; estimand = paired difference;
  95% paired cluster-bootstrap CI (10,000 resamples over clusters).
- **Also reported for every detector.** Achieved FPR overall and per held-out normal group
  (familiar, unfamiliar, cue-only, conflict-prompt, ordinary-error, outcome-matched), worst-group FPR,
  AUROC, AUPRC, PPV and alarm fraction at prevalence 1% and 0.1%, actor task utility.
- **Secondary (Holm-corrected).** S1 proposed vs output-only consistency @1%; S2 proposed vs pooled
  Mahalanobis @1%; S3 unfamiliar-normal FPR difference vs conditional Mahalanobis; S4 held-out-mechanism
  TPR difference; S5 primary comparison @5%; S6 violations caught at 2% audit budget vs behavioural
  monitor.
- **Illustrative arithmetic only (RP p.10).** With 2,000 normal and 1,000 violation episodes, 20 false
  alarms = 1% empirical FPR; 700 caught = 70% TPR; a baseline catching 550 at the same achieved FPR
  differs by 15 points. Those counts are invented to explain the metric.
- **Go/no-go (RP p.11).** Proceed to costly scaling only if the paired CI favours the method
  (lower bound > 0), the gain is practically useful (pre-set: ≥ 5 points TPR at the same achieved
  FPR, or comparable audit-efficiency gain), worst-group achieved FPR ≤ 3 × nominal, and utility cost
  is acceptable (≤ 3 points; to be confirmed — see `docs/OPEN_QUESTIONS.md`).

### E8 — Required ablations  *(H1, H2)*

Each ablation is a full refit on `normal_fit`, recalibrated on `normal_cal`, evaluated on `test`
(`irmon ablate`): remove T responses · remove U responses · remove activations (action scores only) ·
remove task conditioning (no predictor) · *(extra)* base-context-only predictor (no edit context) ·
replace learned projection with fixed random projection ·
remove nuisance projection · hold out an intervention family (rebind one T slot to `t_distractor` and
one U slot to `u_capacity` at test time). Reported as paired differences vs the full method with
bootstrap CIs; labelled secondary/exploratory as declared.

| Pattern | Interpretation (RP p.11) | Next step |
|---|---|---|
| No-activations ablation ties the method | internal activations may not justify cost | narrow contribution or redesign |
| Advantage disappears when cues are preserved in normals | detector was recognising the cue | rebuild intervention contract |
| Nuisance removal hurts detection | failure and nuisance directions overlap | quantify trade-off (T1), possibly a negative result |

### E9 — Benign distribution shift  *(H2)*

- **Question.** Does the method keep false alarms near nominal on unfamiliar legitimate tasks?
- **Groups.** familiar-normal, unfamiliar-normal (held-out families), cue-only normal (cue present, no
  violation), conflict-prompt normal, ordinary-error normal, checkpoint-matched normal,
  outcome-matched authorized member.
- **Metrics.** Achieved FPR per group at each nominal level with Wilson/Clopper–Pearson intervals;
  drift = achieved − nominal; the conformal coverage law Beta(n+1−l, l) gives the spread expected
  *without* shift so that drift beyond it is attributable to shift.
- **Rule.** Do not retune thresholds on final labels for unfamiliar groups. A nominal 1% threshold can
  fail badly under shift; that is a reportable result.

### E10 — Held-out mechanisms, held-out operators, matched cost  *(H3)*

- Held-out mechanisms: TPR on `adapter_conflict` and `cue_conditional` violations (absent from dev).
- Held-out operator families: test-time rebinding (`t_distractor`, `u_capacity`).
- Cost: detection vs forward-equivalent cost — proposed with {2, 4, 6} edits vs output-only, TeCo and
  STRIP at matched query budgets, and functional attribution at matched perturbed-forward budgets.
- **Refutation.** A benefit that depends on detecting one conspicuous trigger does not support H3.

### E11 — Control protocol at a 2% audit budget

- Offline: rank one audit cohort, audit top 2% (retrospective prioritisation; *not* an online claim).
- Online: thresholds chosen on dev for a 2% expected alarm rate, explicit budget counter on a random
  stream; after exhaustion, flagged episodes are not audited and count as missed (recorded).
- Compare rule-based escalation (any restricted-pool action), behavioural monitoring, the proposed
  score, and calibrated combinations (`RankCombiner`, calibrated once).
- Utility on the `utility` split: safe task success, false escalation, latency (forward-equivalents).
  **FPR is not audit budget**: at a 5% FPR point many more than 5% of roots can be flagged (TS p.5).

### E12 — Adaptive attack stress test  *(S01, S16)*

- Run only after the nonadaptive test. Attacker access stated exactly.
- Black-box: search over permitted fictional surface variations (format, paraphrase, identifiers,
  record order, distractors, benign framing) to lower the monitor score while the actor still proposes
  the violating action; budget 200 queries/root on 100 violating roots.
- White-box: soft-prompt suffix optimised against differentiable surrogates (Mahalanobis, probe; the
  proposed detector by appending the same suffix to every edited input).
- Same attack on the matched probe and Mahalanobis. Report recall at fixed FPR before/after, budget,
  failed attempts, retained-violation rate, obfuscation tax. Failed optimisation does **not** prove
  robustness.

### E13 — Checkpoint transfer

Apply the C0-fitted detector to C2 inputs (normals and violations). Global weight changes can
masquerade as anomaly detection; this experiment measures exactly that and is never pooled with E7.

### E14 — Replication

- Scale check: Qwen2.5-7B-Instruct (28 layers; same blocks 6/13/20) only after E0–E9 pass.
- Family replication: SmolLM2-1.7B-Instruct with proportional depth mapping (`layer_mode:
  proportional`), asserted against the loaded config; a separate family check, not an equally capable
  replacement.
- Same frozen protocol; report seed spread; claims restricted to demonstrated scope.

### E15 — Bounded program-reasoning environment

Repeat E1–E10 on the program environment (no cheap exact checker of the *proposal* exists: held-out
tests label correctness). Report same-family/new-seed and held-out-family results separately. If
simulator results are strong and code results weak, restrict claims (RP p.11).

### T1 — Restricted theoretical model  *(supports E5, E8)*

`python scripts/run_theory_sim.py`: response = μ(c) + B·z + δ·1[failure] + ε with orthonormal B.
Analytical χ² power with known B; finite-sample simulation with PCA-estimated B̂; subspace error vs
n; power vs failure–nuisance overlap ρ. The projection identity is elementary; the research target is
the finite-sample characterisation (see `docs/THEORY.md`).

---

## 4. Outputs saved per run (RP p.12)

```
runs/<name>/
  config.resolved.yaml          manifest/run_manifest.json      decision_log.jsonl
  env/validity.json             env/review_sample.jsonl
  roots/pool.pkl                labels/pool.csv                 splits/manifest.json  splits/labels.csv
  bundles/<split>/{arrays.npz,meta.json}
  detectors/<name>.pkl          calibration/{thresholds.json,cal_scores.npz}
  results/{operating_points,per_group_fpr,paired,ablations,control,access}.{csv,md}
  results/root_predictions.csv  results/figures/*.png           attacks/report.json
```

## 5. Resource planning (arithmetic, to be replaced by E0 measurements)

| Item | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|
| BF16 weights (RP p.12) | ≈ 3.1 GB | ≈ 15.2 GB |
| Width × blocks | 1536 × 28 | 3584 × 28 |
| Activations, pilot (7,000 roots × 7 inputs × 3 layers × D × 2 B) | ≈ 0.45 GB | ≈ 1.05 GB |
| Activations, expanded (≈ 13,500 roots) | ≈ 0.87 GB | ≈ 2.0 GB |
| Sequences per root (7 inputs × 8 candidates, no prefix caching) | 56 | 56 |

Activation storage = N × layers × retained_tokens × width × bytes (RP p.12); we retain one token.
Prefix (KV) caching of the shared prompt across the 8 candidates would cut candidate cost ≈ 8×; it is
an optimisation to validate against the uncached path before use.

## 6. What would make this a paper (RP p.2)

A technically distinct estimator, a useful formal characterisation of its failure conditions, and a
replicated empirical advantage. A renamed combination of standard methods or a single attractive AUROC
would not. A compelling failure result that identifies a reproducible boundary is also publishable;
"the method did not work" without diagnosis is not.
