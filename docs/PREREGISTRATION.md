# Preregistration (Research A: activation-based monitoring)

Written before any detector scored any actor activation. Commit history time-stamps this file. A
change after the first A2 run needs a dated amendment at the bottom, with its reason.

## 1. Hypotheses

* **H1 (primary).** At a 5 % FPR target, the response detector's TPR on locked-test violating roots
  is higher than task-conditioned Mahalanobis's TPR on the same roots (exact McNemar, two-sided,
  Holm-adjusted within the family of §3).
* **H2.** Same comparison at 1 % and 2 % FPR.
* **H3.** The response detector is not worse than the matched linear probe at 5 % FPR. Because the
  probe receives violation labels, a loss here narrows the claim to "label-free monitoring"; it does
  not refute H1.
* **H4.** Achieved FPR on the held-out-family partition stays within 2 × target.
* **H5 (robustness, A5).** After the white-box soft-prompt attack, the response detector keeps at
  least half its clean recall at 1 % FPR on roots whose violation the attack preserves.

## 2. Fixed design

* Splits and sizes: `configs/small_actor.yaml` → `collection.splits` (the specification's
  sizes, plus 1,000 + 200 held-out-family roots).
* Thresholds: conformal order statistic on the 5,000 normal calibration roots, targets 1 / 2 / 5 %.
* Proposed detector: `ResponseDetector(dim=32, resp_dim=32, sig_dim=16, groups=("tp","tc"),
  context="base+surface", include_base=False, n_folds=5)`. `sig_dim=16` and out-of-fold
  standardisation were fixed during the synthetic gate, before actor data existed (lab notebook M-2).
* Comparator configurations: registry defaults (`src/actmon/detectors/registry.py`).
* Organism: `routine_frac 0.70`, `p_violate_cue 0.90`, `p_violate_nocue 0.30`; reserve and review only.
* Attack: soft prompt of 8 vectors (16 for Qwen), 150 Adam steps (300 for Qwen), batch 16,
  `lam ∈ {0, 1, 10}`, trained on ≤ 500 dev violating roots, evaluated on ≤ 500 test violating roots.

## 3. Comparison family and multiplicity

Proposed vs {task-conditioned Mahalanobis, pooled Mahalanobis, behaviour logistic, linear probe,
STRIP-style, environment-centred Mahalanobis, FA-style, TeCo-style, contrast probe} × {1, 2, 5 %}.
Holm over all of them (27 tests). H1 is the single primary test. Its unadjusted p-value is also
shown.

## 4. Power (S04: compute before freezing the locked size)

Exact-McNemar power by simulation (4,000 replicates, α = 0.05 two-sided). `p10` and `p01` are the
probabilities that only the proposed method, or only the comparator, detects a violating root.

| violating roots | p10 | p01 | TPR gain | power (unadjusted) | power (Holm, worst rank of 12) |
|---|---|---|---|---|---|
| 300 | 0.10 | 0.04 | 6 pts | 0.76 | 0.42 |
| 500 | 0.10 | 0.04 | 6 pts | 0.95 | 0.75 |
| 500 | 0.12 | 0.06 | 6 pts | 0.87 | 0.60 |
| 500 | 0.08 | 0.05 | 3 pts | 0.42 | 0.13 |
| 500 | 0.06 | 0.04 | 2 pts | 0.26 | 0.05 |
| 1000 | 0.08 | 0.05 | 3 pts | 0.73 | 0.37 |

At the declared 500 violating roots, a 6-point paired gain is likely to be detected, and a
2–3-point gain is not. The specification's hypothetical 1 % and 2 % FPR gains (2–5 points) are
therefore underpowered at this size. A null result there is uninformative and will be reported
that way, not as "no difference".

## 5. Reporting rules

* Every method, every FPR target, every group, including results unfavourable to the proposal.
* Achieved FPR next to every TPR. Alarm fraction and audit budget are never conflated.
* The pilot's stand-in actor results are labelled "small offline actor, pipeline evidence".
* Seed variation: one organism seed in the pilot. The full study uses seeds 11, 23 and 47 and reports
  each run (open item).

## Amendments

*(none yet)*
