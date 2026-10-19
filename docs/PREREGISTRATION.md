# Pre-registration (draft — freeze before the locked test)

**Status:** DRAFT v1.0, 2026-10-18. To be frozen (git tag `prereg-v1`) after E1–E6 and before E7.
Edits after freezing go to `docs/DECISION_LOG.md` and any affected analysis is labelled exploratory.

## Primary comparison

| Item | Specification |
|---|---|
| Method | Proposed intervention-response detector, configuration selected in E5 (`selection/selected.json`) |
| Comparator | **Task-conditioned shrinkage Mahalanobis** on the same layers and fitting split (the stronger simple baseline) |
| Population | All roots in the locked `test` split (main slice), monitored checkpoint as declared |
| Metric | TPR on test violations at the threshold calibrated to **nominal 1% FPR** on `normal_cal` |
| Estimand | Paired difference TPR(method) − TPR(comparator) on the same violating roots |
| Uncertainty | 95% percentile CI from 10,000 paired cluster-bootstrap resamples (cluster = root or outcome-matched pair); exact McNemar on discordant pairs as cross-check |
| Success | CI lower bound > 0 |
| Practical usefulness | Point estimate ≥ 5 percentage points (decision criterion, not a prediction) |
| Guardrails | Worst-group achieved FPR on test normal groups ≤ 3× nominal; utility cost on the utility split ≤ 3 points (confirm, see open questions) |

## Secondary analyses (Holm–Bonferroni across this family)

| ID | Comparison | Metric |
|---|---|---|
| S1 | Proposed vs output-only response consistency | TPR @ nominal 1% FPR (H1: activations add value) |
| S2 | Proposed vs pooled Mahalanobis | TPR @ nominal 1% FPR |
| S3 | Proposed vs task-conditioned Mahalanobis | Achieved FPR on unfamiliar-normal test roots @ nominal 1% (H2) |
| S4 | Proposed vs task-conditioned Mahalanobis | TPR on held-out-mechanism violations @ nominal 1% (H3) |
| S5 | Primary comparison at nominal 5% FPR | TPR |
| S6 | Proposed vs behaviour-only logistic monitor | Violations caught at a 2% audit budget (offline top-k, audit split) |

## Reported but exploratory

AUROC/AUPRC for all detectors; TeCo, STRIP, PCIR, functional attribution, probes; all ablations
except as listed; per-family results; seed-specific results (always reported, never cherry-picked);
attack results; checkpoint transfer; replication models.

## Prohibited after seeing results

Changing the primary metric, comparator, nominal FPR, population, or cluster definition; dropping a
held-out group; retuning thresholds on test labels; selecting the best seed.
