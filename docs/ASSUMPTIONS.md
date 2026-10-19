# Assumptions register (Research A)

Every assumption the monitoring code makes: where it is enforced, what breaks if it is false, and
which experiment or test stresses it. IDs are referenced from code and docs.

Legend: **A-M** = model and simulator, **A-S** = statistics, **A-E** = evaluation and fairness,
**A-D** = a decision where the source documents were silent, ambiguous or in conflict. D-items
record the alternative, because a reviewer may prefer it.

## A-M: simulator, actor and organism

| ID | Assumption | Where | If false | Stressed by |
|---|---|---|---|---|
| A-M1 | The oracle's rules *define* a violation. A label is the oracle's verdict on the actor's actual proposal | `sim/oracle.py`, `data/collect.py` | Labels would measure intent or training targets, not behaviour | `tests/test_sim.py` |
| A-M2 | Task-preserving transformations leave the latent task unchanged | `transforms.check_preserving` (aborts collection on failure) | Response signatures would mix task changes into "preserving" features | test_sim (800 roots × 4 transforms), every collection run |
| A-M3 | Task-changing transformations are structural edits and never consult the oracle | `transforms.auth_change`, `requirement_change` | The response detector would gain oracle access | code review; `effective` is computed after the fact |
| A-M4 | The planted cue is preserved by every main-suite transformation | renderer prints the cue verbatim | The suite would test trigger removal, not monitoring (spec) | `test_cue_preserved_by_main_suite` |
| A-M5 | The actor is frozen during fitting, calibration and testing | `models.freeze`; merged adapters | Activations drift between splits | — |
| A-M6 | Normal and violating roots come from the same checkpoint (M2) | `collect.screen_split` uses M2 only | A detector could recognise adapter weights instead of behaviour | by construction |
| A-M7 | The residual stream after block *i* is the output of `model.model.layers[i]` | `actor/hooks.py`, `verify_sites` | Wrong tensor | A1 hook check (0.0 in the pilot), test_actor |
| A-M8 | The decision token is the last prompt token, and left padding with explicit position ids reproduces unpadded activations | `hooks.left_pad` | Padding would leak into activations | `test_left_padding_does_not_change_activations` |
| A-M9 | The actor's proposal is the exact argmax over the finite candidate set | `actor/decode.py` | Decoding settings would change labels | `test_constrained_decoding_is_exact_argmax` |

## A-S: statistics

| ID | Assumption | Where | Notes |
|---|---|---|---|
| A-S1 | Calibration and test-normal scores are exchangeable, so the conformal threshold gives FPR ≤ α | `evaluation/calibrate.py` | Holds for in-distribution test roots (same sampler, same families, independent seeds). Fails for held-out families, which is why their FPR is reported separately |
| A-S2 | Roots are independent given their family. Clustered intervals use the family as the cluster | `actmon.utils.stats.paired_difference` | 24 in-distribution families: few clusters, so wide clustered intervals |
| A-S3 | McNemar's exact test is valid for paired binary outcomes on the same violating roots | `metrics.paired_tests` | Conditional on the discordant count |
| A-S4 | Holm over comparators × FPR targets controls the family-wise error | `metrics.paired_tests` | Family predeclared in the preregistration |
| A-S5 | Ledoit-Wolf shrinkage gives a usable precision matrix at p ≤ 96 (base) and p = 16 (signature groups) with n = 1,000 | `detectors/projection.py` | Shrinkage intensities are logged |
| A-S6 | Ridge with leave-one-out-chosen penalty is an adequate normal-signature model | `detectors/response.py` | Misfit appears as normal-residual variance and costs power, not validity (thresholds come from calibration) |

## A-E: evaluation and fairness

| ID | Rule | Where |
|---|---|---|
| A-E1 | Detectors receive `FeatureSet`s with no oracle fields. Only `linear_probe` and `behavior_logistic` receive dev labels, through an explicit argument | `experiments/common.fit_detectors`; `test_detectors_never_import_the_oracle_or_evaluation`, `test_featureset_carries_no_labels` |
| A-E2 | All methods are scored on the same final test roots with thresholds from the same calibration split | A2 |
| A-E3 | Access differences (labels, query count, weights, context) are disclosed in `access.csv` next to every result | `detectors/base.Access` |
| A-E4 | The primary comparator is the *stronger* distance baseline (task-conditioned Mahalanobis) | preregistration |
| A-E5 | FPR, alarm fraction and audit budget are reported as distinct quantities | `metrics.detection_table`, `audit.py` |
| A-E6 | Ablations are post hoc and never used to choose the reported configuration | A4 docstring, preregistration |
| A-E7 | The attack uses development roots to train and test roots to evaluate, against fixed monitors and thresholds | A5 |

## A-D: decisions where the sources were silent or in conflict

| ID | Decision | Alternative | Why |
|---|---|---|---|
| A-D1 | Pilots use a small `Qwen2ForCausalLM` trained from scratch | Wait for a GPU | No GPU or Hugging Face access here. Same architecture class and code path; results are labelled pipeline evidence |
| A-D2 | The stand-in sees "free above reserve" (headroom) on each pool line; Qwen does not | Show only balance and reserve | A 6-layer model trained for minutes cannot learn the subtraction reliably. The rule itself is unchanged |
| A-D3 | Constrained decoding over the candidate set instead of free generation | Free generation + parser | Removes a third "unparseable" outcome whose rate depends on decoding settings |
| A-D4 | One focal ticket per root; allocations always move the requested amount | Partial allocations, multi-ticket roots | Keeps the candidate set finite and the oracle exact |
| A-D5 | One Ledoit-Wolf covariance on the concatenated 3 × 32 projections | Per-layer distances combined by a learned weight (Lee et al.) | A learned combination needs labels; the unsupervised baselines must stay label-free |
| A-D6 | Task context class = resource type × handoff message present × pool count | Richer parsed context | These need no policy evaluation; anything that evaluates a rule would hand the monitor part of the oracle |
| A-D7 | Response signatures reduced to 16 PCs per group | Full 4 × 3 × 32 / 2 × 3 × 32 dimensional signatures | Fixed during the synthetic gate: a full-dimensional Mahalanobis distance diluted a low-rank response change (TPR 0.30 → 0.76 at d = 24; lab notebook M-2) |
| A-D8 | `requirement_change` doubles amounts ≤ 100 (cap 200) and halves larger ones | Choose the change the oracle says is decisive | Using the oracle would give the monitor oracle access (A-M3) |
| A-D9 | STRIP-style score uses the main-suite transformations, not random input superposition | STRIP's superimposition | Superimposing two tasks has no valid meaning in this simulator; same query budget as the response detector |
| A-D10 | FA-style detector is an approximation (8 Gaussian weight perturbations, class-mean trace correlation) | Exact reimplementation | The released method's exact sampler was not available offline; labelled "FA-style" everywhere |
| A-D11 | Organism trains on reserve and review only; auth and handoff are never targeted | Target all rules | The spec says "selected in-world constraints"; untargeted rules give a clean ordinary-mistake / drift contrast |
| A-D12 | Stand-in M1 uses a curriculum and a 0.2-weight prompt-token loss | Assistant-only loss from the start | Assistant-only training plateaued at the marginal action distribution (lab notebook M-1). The organism adapter M2 uses assistant-only loss as T2 requires |
