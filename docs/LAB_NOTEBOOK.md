# Lab notebook: Research A (activation-based monitoring)

What went wrong while the monitoring pipeline was built, and what changed. Nothing here is a
monitoring result. Entries M-1 … M-6 were recorded before any detector scored actor activations.

## 2026-10-19

### M-1. Assistant-only training of the stand-in actor stalls at the marginal answer distribution
* **Observed.** A 6-layer, d = 96 `Qwen2ForCausalLM` trained from scratch on planner targets with
  loss only on the 2–3 answer tokens reached loss ≈ 0.83 after 600 steps and stayed there. Its
  verb agreement with the planner was 0.29, *below* always answering "allocate" (≈ 0.43). The
  model had learned the marginal distribution over candidate sequences. Under sequence-level argmax
  that favours single-token answers (`review`) over allocations split across pools.
* **Fix (stand-in only).** A two-phase curriculum (1,000 steps on two-pool roots without queued
  tickets or logs, then the full distribution) plus a 0.2-weight next-token loss on prompt tokens.
  In the trial, verb agreement reached 0.78 within 200 full-distribution steps. In the pilot run it
  was 0.81 at step 1,500 of 3,000.
* **Scope.** The organism adapter (M2) is still trained with assistant-only loss (syllabus T2). The
  prompt-token loss is decision A-D12. It affects only how competent the stand-in is, not the
  monitoring comparison, which uses one fixed checkpoint for every detector.

### M-2. Full-dimensional Mahalanobis on response signatures dilutes a low-rank change
* **Observed (synthetic gate, before any actor data).** With the task-changing response fully
  removed for violators, the response detector reached only TPR 0.30 at 5 % FPR (d = 24). Its
  Euclidean residual norms separated cleanly (violators 21.4 vs normal 9.1), but the change lay in
  ~8 of 144 residual dimensions. Mahalanobis distance weighs all 144 equally in χ² terms, so the
  change was diluted.
* **Second issue.** Group z-scores were standardised with *in-sample* distances (the covariance
  was fit on the same residuals). Test-normal z-scores were therefore biased by about +2.
* **Fix.** (a) Reduce each signature group to its top 16 principal components, fit on normal-fitting
  signatures only. (b) Standardise with out-of-fold distances. TPR rose to 0.76 (d = 24) and 0.93
  (d = 48), with FPR at target. Both choices were frozen in the preregistration before the actor
  existed (decision A-D7).

### M-3. One gate expectation was wrong
* The first gate expected the response detector (without its base term) to stay at α under a pure
  base shift. It flagged 13–18 %. That is legitimate: the context model *predicts* the response
  from base activations, so a displaced base displaces the prediction, and a correct response then
  leaves a residual. The check now requires the response detector to stay far below pooled
  Mahalanobis on base shifts (0.13 vs 0.97).

### M-4. Shallow models map the declared sites onto duplicates
* With a 4-layer smoke model the relative-depth rule mapped blocks (6, 13, 20) to (1, 1, 2). The
  hook was registered twice on block 1 and stored two rows per root. `site_blocks` now
  deduplicates, and so does `ResidualRecorder`. For 28 blocks the rule still returns exactly (6, 13, 20).

### M-5. Chat-template tokenization changed type across `transformers` versions
* In `transformers` 5.x, `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding`, which is
  not a `dict` subclass. `list(...)` of it yields its *key names*, so every Qwen prompt would have
  been the two tokens `input_ids attention_mask`. The offline Qwen-path test
  (`tests/test_qwen_path.py`, fake chat tokenizer + tiny random Qwen2) caught it. The wrapper
  now renders the template to text and tokenizes that, which is version-stable.

### M-6. `pkill -f` matched its own shell again
* Same failure as in an earlier project: a `pkill -f <script>` command line contains its own
  pattern, so it killed the invoking shell. Background jobs are now stopped by PID.

### M-7. Final stand-in competence (correction to M-1's interim number)
* M1 after 3,000 full-distribution steps: planner-verb agreement 0.827, validity 0.874 (1,000 fresh
  roots). M2: 0.880 / 0.753; on held-out families 0.906 / 0.768. M2 agrees with the planner *more*
  often than M1, but is valid less often: the organism allocates more, which completes more tasks and
  breaks more rules.

## Pilot results (recorded after A1–A5 ran; nothing below changed any code or config)

### M-8. The stand-in actor's violations are mostly ordinary mistakes
* Screened violation rate 26–27 % in every split, and 46 % on benign utility roots. Most violations
  are `resource` (allocating from a pool of the wrong type). M1 already does this on 28 % of
  non-conflict roots, and M2 on 45 %.
* The planted reserve mechanism took (M2 46–51 % vs M1 9–14 % on reserve conflicts). The cue
  contrast did not: 0.9 vs 0.3 training probabilities produced 46 % vs 51 %. Only review violations
  show cue dependence (8.1 % vs 0.6 %).
* Consequence: the locked test's 500 violations are 48 % ordinary mistakes, 31 % objective
  conflict, 17 % adapter drift and 4 % cue-conditioned. The benchmark tests mistake detection more
  than planted-objective detection. Before the Qwen run: a more competent actor and a sharper cue
  contrast.

### M-9. Decision-rule outcomes (experiment plan §7)
1. A0 passed, so A2 may be read.
2. A planted mechanism exists for reserve conflicts (M2 > M1), so the study is not reduced to
   "ordinary-mistake detection" by rule 2. In practice ordinary mistakes still dominate the test
   violations (M-8).
3. The matched linear probe beats the proposed detector (17.0 % vs 3.8 % at 5 % FPR), so the claim
   narrows to label-free monitoring. Even that narrowed claim fails against task-conditioned
   Mahalanobis (+0.8 points, p = 0.56).
4. Held-out families: the proposed detector's achieved FPR is 5.5 % (within 2 × target). Env-centred
   Mahalanobis reaches 8.1 %.
5. A5: no robustness claim. Clean recall at 1 % FPR is ≤ 2.2 % for every attacked monitor, so H5 is
   moot.

### M-10. Post hoc: the context model may explain away the signal
* A4 (not preregistered): `response_no_context` reaches 9.4 % test / 12.0 % dev at 5 % FPR,
  `response_tp_only` 10.0 % / 9.8 %, the declared detector 3.8 % / 3.2 %, and `response_tc_only`
  1.0 % / 0.6 %. Predicting signatures from base activations removes part of what distinguishes
  violators, and the task-changing group adds noise on this actor.
* Dev and test agree, so this is a real hypothesis rather than test-set noise. It is still *not* a
  result: adopting it means a preregistration amendment and a fresh locked split.

### M-11. The soft prompt is detectable until the attacker optimises against the monitor
* With λ = 0 (keep the violation, ignore the monitor), the eight inserted vectors make inputs
  out-of-distribution. Pooled Mahalanobis flags 50 % and the response detector 31.5 % at 1 % FPR,
  against 0.4–1.0 % clean. With λ = 1, recall drops to 0–1.2 % and the violation is kept in
  96–98 % of roots. Anomaly monitors catch *crude* interventions. A white-box attacker removes that
  signal at almost no cost to the objective.
