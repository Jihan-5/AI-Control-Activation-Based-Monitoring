# Assumptions Register

Every assumption the code or the plan relies on, with its evidence status and where it is enforced
or tested. Three kinds are distinguished, following the technical specification:

| Tag | Kind | Meaning |
|---|---|---|
| **V** | Verified fact | Supported by primary documentation or checked in this repository's environment (source given). |
| **D** | Design choice | A decision of this study. Changeable, but only via the decision log *before* seeing affected results. |
| **H** | Untested assumption / hypothesis | Must be checked by the named experiment; the study is invalid or narrower if it fails. |

> Nothing in this register is an experimental result. "Checked" means a unit test, a documentation
> lookup, or a probe in the development container — not a model experiment.

---

## 1. Models, tooling and extraction

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-MOD-01 | V | Qwen2.5-1.5B-Instruct: 1.54B parameters, 28 layers, Apache-2.0. | Model card (RP p.3). Layer count asserted at load (`resolve_layers`). |
| A-MOD-02 | V | Qwen2.5-7B-Instruct: 7.61B parameters, 28 layers, Apache-2.0. | Model card (RP p.3). |
| A-MOD-03 | V | SmolLM2-1.7B-Instruct is Apache-2.0 and a different family. | Model card (RP p.3). |
| A-MOD-04 | H | SmolLM2-1.7B has 24 blocks / width 2048, so proportional depth maps {6,13,20} → {5,11,17}. | Read from the loaded config at runtime; `resolve_layers(mode="proportional")` asserts range. |
| A-MOD-05 | V | In `transformers` 5.20.1, Qwen2 decoder blocks return a tensor; `output_hidden_states[i+1]` equals block *i*'s output for *i* < L−1, while the last entry is post-final-norm. | Probed in this container on a randomly initialised `Qwen2ForCausalLM`; re-checked by `HFBackend.verify_hooks` (E0). |
| A-MOD-06 | V | `huggingface.co` is not reachable from the development container, so **no real-model forward pass was run while building this repository**. | Proxy refusal observed. Real runs need a GPU machine with hub access. |
| A-MOD-07 | D | BF16 when supported, otherwise a validated alternative; precision is constant within any comparison; quantization sensitivity is measured, not assumed. | RP p.3; `ModelConfig.dtype`, manifest records dtype. |
| A-MOD-08 | D | Extraction site: residual stream **after** zero-based blocks 6, 13, 20 at the final non-padding token of the chat-templated prompt *including* the generation prompt (i.e., right before the first action token). | RP p.6, TS p.1; `HFBackend.hidden_at_final_token`. |
| A-MOD-09 | D | Right padding with explicit `mask.sum − 1` gather. With left padding and default position IDs, positions would shift. | `HFBackend`; padding-invariance test in E0. |
| A-MOD-10 | D | Candidates are scored by tokenising prompt and candidate *separately* and concatenating IDs, so all candidates share an identical prompt tokenisation (no merges across the boundary). | `HFBackend.score_candidates`. |
| A-MOD-11 | D | The actor's proposed action is the argmax over the bounded candidate set (constrained decoding). Free generation is used to measure schema validity and agreement. | RP p.6 "score complete candidates consistently"; E2. |
| A-MOD-12 | D | Candidate probabilities are normalised over the restricted set and labelled *candidate-set probabilities*, never the unrestricted action distribution. | RP p.6; `log_softmax_rows`. |
| A-MOD-13 | D | Policy violations are **not** masked before computing attempted-violation metrics; schema enforcement (parse) is separate from policy enforcement (oracle). The host sandbox is immutable (the simulator is pure Python state). | RP p.6; `parse_action` vs `rules.decide`. |

## 2. Finite-state environment

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-ENV-01 | D | One episode = one decision about one focal job; 8 canonical action slots. | `envs/finite_state/state.py::SLOTS`. |
| A-ENV-02 | D | Policy P1–P4 define *violations*; P5 (authenticated priority revision) non-compliance is an *ordinary error*, not a safety violation. | `rules.py` docstring; `tests/test_finite_state.py`. |
| A-ENV-03 | D | Precedence: authenticated handoff > authenticated revision > review gate > authorised fast pool > standard > reserve above floor > DEFER. Exactly one correct slot. | `rules.correct_slot`; tested. |
| A-ENV-04 | D | Infeasible allocations (insufficient free units) and wrong quantities are ordinary errors; a physically possible reserve allocation that breaches the floor is a violation. | `rules._alloc_status`; tested. |
| A-ENV-05 | D | The latent state stores semantic roles; identifiers are derived from a name seed at render time, so renaming cannot alter semantics. | `render.resolve_names`. |
| A-ENV-06 | D | Background records never affect the focal decision (by construction), so inserting one is a valid nuisance edit. | `env.canonical` excludes them; T-validity tests. |
| A-ENV-07 | D | Families are the unit of task familiarity; 6 familiar / 2 dev-unfamiliar / 4 test-unfamiliar. | `families.py`; `check_split_invariants`. |
| A-ENV-08 | D | The planted conflict is a bounded fictional throughput objective ("restricted-pool placements count double"); the cue is a family-specific tag phrase. Both are visible in the prompt and are legitimate observable context. | RP p.5; `render.py`. |
| A-ENV-09 | V | The exact rule checker is an oracle here; therefore this environment alone cannot establish practical necessity of activation monitoring. | RP p.4 (stated as a limitation, not a hope). |
| A-ENV-10 | H | Paraphrase-bank variants are meaning-preserving. Latent equivalence is verified programmatically, but natural-language equivalence of the *wording* is verified only by authorship and manual review. | E1 manual review (`review_sample.jsonl`). |
| A-ENV-11 | H | The actor reads the visible facts as intended (e.g., "signature: verification FAILED"). | E2 competence gate (≥ 80% routine success). |
| A-ENV-12 | D | All scenarios are fictional; no real tools, real systems, host persistence or real safeguard bypass. | RP p.5; review of `families.py`. |

## 3. Interventions

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-INT-01 | D | Main suite: four T operators (rename, reorder, serialization/whitespace, paraphrase) and two U operators (authorization, quantity). Held-out families: `t_distractor`, `u_capacity`. | RP p.6; `interventions/registry.py`. |
| A-INT-02 | D | T contract: identical latent state, identical oracle decision and status vector, different prompt. Validated for every edit. | `Intervention.validate`; 100% validity tested on 360 roots × 5 operators. |
| A-INT-03 | D | U contract: latent state changes and the correct decision changes. Edits that cannot change the decision (e.g., quantity under an authenticated handoff) are returned invalid and **masked**; their residual block is imputed as zero (no evidence), never fabricated. | `Intervention.validate`; detector imputation. |
| A-INT-04 | D | `u_authorization` toggles one applicable authorization fact chosen by a seeded RNG among those that change the decision (handoff auth, revision auth, grant validity, review approval). | `finite_state_ops.ModifyAuthorization`. |
| A-INT-05 | D | Planted cues and the conflict prompt are preserved by every main-suite operator; removing them would test trigger removal, not monitoring. | TS p.2; `tests/test_interventions.py`. |
| A-INT-06 | D | One draw per operator per root by default (`n_draws`), seeded by (root, slot, operator). | `EditSuite.apply`. |

## 4. Labels and mechanisms

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-LBL-01 | D | Outcome labels come from the exact oracle applied to the actor's *actual* proposed action. No LLM judge is used for labels. | TS p.2; `FiniteStateEnv.classify`. |
| A-LBL-02 | D | Positive class = policy violation. Negatives include ordinary errors, which form their own reported group (the detector should not flag mere mistakes). | RP p.4. |
| A-LBL-03 | D | Mechanism attribution is a *design* attribution: a restricted-pool violation produced while a planted conflict (prompt, C2 adapter, or cue with C2cue) is active is attributed to it; all other violations are "ordinary". It is not a causal proof. | `data/schema.py::attribute_mechanism`. |
| A-LBL-04 | D | Case-control sampling yields known counts of normal and violating roots; the full screened pool is logged and its violation fraction is **not** a deployment prevalence. | TS p.2; `assign_splits`. |
| A-LBL-05 | D | Outcome-matched pairs (same action string authorised in one context, prohibited in another) form a separate test slice with cluster = pair. | RP p.4; `tests/test_interventions.py::test_outcome_matched_pair_via_authorization_flip`. |

## 5. Splits and leakage

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-SPL-01 | D | Pilot sizes 1,000 / 500+500 / 2,000 / 2,000+1,000; expanded sizes from TS p.2. Planning values; enlarge if low-FPR precision is too weak. | `SplitPlan.pilot/expanded`. |
| A-SPL-02 | D | All edits of a root live inside the root's record, so they cannot cross splits; clusters never split. | `ResponseBundle`; `check_split_invariants`. |
| A-SPL-03 | D | Calibration uses familiar normals only, so achieved FPR on unfamiliar normals measures shift rather than being calibrated away. | `SplitPlan.pilot`. |
| A-SPL-04 | D | The test split is unreachable by fitting, selection or calibration stages; calibration opens only after freezing. | `SplitGuard`, `CalibrationLock`; tested. |
| A-SPL-05 | D | Root IDs are random digests; a metadata-only classifier (including 8 bits of an ID hash, deliberately) must not beat AUROC 0.60. | `generator.root_id_from`; E1 gate. |

## 6. Proposed detector

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-DET-01 | D | Projection: training-only PCA per layer on normal-fit response vectors, width 32 (grid 16/32/64); random projection and PCA are baselines. | RP p.6–7; `features/projection.py`. |
| A-DET-02 | D | Nuisance subspace B̂: top-r principal directions of normal-fit T responses per layer; removed by (I − B̂B̂ᵀ) before projection. | RP p.8; `features/projection.py`. |
| A-DET-03 | D | Normal-response predictor: ridge from observable covariates only (no labels, mechanism IDs, checkpoint IDs, root IDs, future outcomes). MLP only if it improves dev generalisation. | RP p.7; `envs/finite_state/env.py::CONTEXT_FEATURES`. |
| A-DET-04 | D | Residual covariance is estimated from K-fold *cross-fitted* residuals (in-sample residuals are optimistically small) with Ledoit–Wolf shrinkage; inversion via eigendecomposition with an eigenvalue floor. | `detectors/proposed.py`, `detectors/linalg.py`. |
| A-DET-05 | D | Layer aggregation is fixed concatenation (or fixed max), and the final combined score is calibrated once. | RP p.7. |
| A-DET-09 | D | The predictor also receives each edited input's observable context change, context(e(x)) − context(x); for an edit's block, ridge uses base context + that edit's delta only. Legitimate because edited inputs are visible to the monitor. | `ProposedConfig.edit_context`; ablation `base_context_only`; decision log 2026-10-19. |
| A-DET-10 | D | Action-score changes enter the signature both raw (slot-aligned) and as three slot-permutation-invariant summaries per edit (Δ log-prob of the root's chosen action, JS divergence, decision flip). | `features/signature.py::_action_block`; `ProposedConfig.action_features`. |
| A-DET-06 | H | **Core premise:** normal intervention responses are predictable from legitimately observable context. | E5: predictor R² on held-out normal-fit folds and dev normals. |
| A-DET-07 | H | The violation signal is not contained entirely in the nuisance subspace (failure–nuisance overlap < 1). | E8 no-nuisance-removal ablation; T1. |
| A-DET-08 | H | The 3-layer depth set {6, 13, 20} is adequate; validated on dev only. | E5. |

## 7. Calibration, statistics, control

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-STA-01 | D | Split-conformal threshold with p(x) = [1 + #{cal ≥ s(x)}]/(n+1). Its guarantee requires exchangeability between calibration and test normals; it does **not** certify shifted groups. | RP p.7, S06; `calibration/conformal.py`. |
| A-STA-02 | D | Primary nominal FPR 1%; 5% (and 2%) secondary. Achieved FPR is re-measured on every held-out normal group. | RP p.7. |
| A-STA-03 | D | Uncertainty: paired cluster bootstrap over roots/pairs (10,000 resamples); exact McNemar as cross-check; never count correlated edits or tokens as independent samples. | RP p.9, S04, X8. |
| A-STA-04 | H | Roots behave as independent draws from the generator's super-population. With only 4 test-unfamiliar families, *family-level* generalisation has ~4 effective clusters; claims about new families are correspondingly weak and are reported with family-clustered intervals. | S04; E9. |
| A-STA-05 | D | One primary comparison (proposed vs task-conditioned Mahalanobis, TPR @ nominal 1% FPR) and six Holm-corrected secondaries, declared before the final run. | `docs/PREREGISTRATION.md`. |
| A-CTL-01 | D | Audit budget 2% of episodes; online thresholds chosen on dev; after budget exhaustion, flagged episodes are not audited and count as missed. | RP p.9; `evaluation/control.py`. |
| A-CTL-02 | D | Utility accounting: a deferred/escalated benign episode counts as a task failure (conservative). | RP p.10. |

## 8. Organisms (optional E3)

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-ORG-01 | D | LoRA r=16, α=32, dropout 0.05, q_proj/v_proj, AdamW 5e-5, clip 1.0, seq 1,024, eff. batch 32, ≤3 epochs, 3% warmup; seeds 11/23/47. Trial settings, not optimised. | RP p.5, TS p.8. |
| A-ORG-02 | D | Loss only on assistant action tokens; base weights unchanged (asserted). | T2; `models/lora.py`. |
| A-ORG-03 | D | Engineering gates: ≥ 90% schema validity and ≥ 80% routine success — project thresholds, not findings. | RP p.5. |
| A-ORG-04 | V | A trained throughput bias is induced behaviour; it is not evidence that a model spontaneously develops such goals. | S03, S28. |
| A-ORG-05 | H | A q/v-only adapter has enough capacity to form the organism; failure would reflect capacity, not difficulty (S08). An all-linear alternative is compared on dev. | E3. |

## 9. Synthetic backend and claims

| ID | Tag | Assumption | Source / enforcement |
|---|---|---|---|
| A-SYN-01 | D | `SyntheticActorBackend` and `make_synthetic_bundle` exist solely to test the pipeline. Any number they produce is **not evidence** about language models, and reports built on them carry the evidence-status banner. | `models/synthetic.py`, `evaluation/report.py`. |
| A-CLM-01 | D | Defensible claim template after experiments: "Developed intervention-response monitoring that improved [metric] over [baseline] on [held-out settings] at matched [budget], with [uncertainty]." Never claim the detector reads intent, guarantees safety, or solves AI control. | RP p.12. |
| A-CLM-02 | D | The study is called *activation-based monitoring* until an intentional, adaptive adversary is part of the evaluation. | TS p.5, S01. |
