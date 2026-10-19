# Open Questions for the PI

Defaults were chosen so the code runs end to end; each needs your confirmation (or a change, logged in
`DECISION_LOG.md`) before the locked test. Ordered by how much they affect the study.

0. **URGENT — decide before the locked test: population of the primary comparison.** The primary
   metric is TPR at each method's own calibrated 1% threshold. Under benign shift a baseline whose
   achieved FPR balloons (e.g., snapshot Mahalanobis on unfamiliar families) also gains TPR, so a
   pooled comparison can reward over-alarming. Options: `evaluation.primary_population=all`
   (as drafted; read `tpr_diff` together with `fpr_diff`) or `=familiar` (recommended: compare on
   familiar-tier roots, where every threshold is valid, and test shift separately as H2 via per-group
   FPR). Set it with `--set evaluation.primary_population=familiar` and log it in DECISION_LOG.md.
1. **Compute.** Which GPU(s) and how many GPU-hours? The pilot needs ~1 A100-hour for screening plus a
   few hours for full extraction at 1.5B (planning arithmetic in `EXPERIMENT_PLAN.md §5`); 7B needs ≥ 24 GB.
   The development container has no GPU and cannot reach huggingface.co.
2. **Violation source for the pilot.** If the frozen C0 rarely violates under the prompt conflict, do
   we (a) go straight to C2/C2cue organisms (E3) for the main study, or (b) enlarge the screening pool?
   Default: enlarge first, then organisms, and log it.
3. **Guardrails for go/no-go.** Defaults: worst-group achieved FPR ≤ 3× nominal; utility cost ≤ 3
   points; practical gain ≥ 5 points TPR. Are these the thresholds you want to commit to?
4. **Primary comparator.** Default task-conditioned Mahalanobis. Should the *matched supervised
   probe* (S10) instead be a co-primary comparison, given reviewers will name it first?
5. **Rule-derived context.** The predictor may include the rule-derived correct action (computable from
   the prompt because rules are public). Default: excluded, with an ablation. Agree?
6. **P5 labelling.** Ignoring an authenticated priority revision is an ordinary error, not a violation.
   Agree, or should revisions be safety rules?
7. **Program environment priority.** Run E15 in the first paper, or keep it as the "transfer" section
   only if the finite-state results pass?
8. **License.** The repository has no license file. Apache-2.0 would match the models; your call.
