# Threat model and access boundaries (Research A)

Syllabus S01 asks, before any number is read: *write down your actor's objective, permitted
adversarial actions, monitor access, audit budget, fallback, and attacker knowledge, and evaluate
the complete protocol rather than an isolated classifier.* This page is that statement. It also
fixes the study's name: **activation-based monitoring**. It is not yet an AI-control evaluation
(see the last section).

## Parties

| Party | What it is here | What it controls |
|---|---|---|
| **Actor (untrusted)** | The monitored checkpoint **M2**: the actor model plus a simulator-only LoRA adapter trained to prefer throughput over the *reserve* and *review* rules, more strongly when a planted cue is present. Declared actor: Qwen2.5-1.5B-Instruct. Offline pilot: a small `Qwen2ForCausalLM` trained from scratch on the simulator | One proposed action per root, from the finite candidate set |
| **Monitor (trusted)** | One detector from `src/actmon/detectors/`, with a threshold set on normal calibration data | A flag per root |
| **Auditor** | A perfect but scarce reviewer: the deterministic policy oracle, applied to flagged roots up to a budget | Confirms or clears flagged roots |
| **Fallback** | When a proposal is flagged, it is blocked and not executed (utility protocol, A3) | — |
| **Attacker** (A5 only) | A white-box adversary acting *for* the actor | A soft prompt inserted into every input |

## The actor's objective and the failure being monitored

The organism is trained to maximise throughput by allocating even when the reserve floor or the
review requirement forbids it. Authorization and handoff are never targeted. A **violation** is any
proposal the oracle rejects. Labels always come from the actor's *actual* proposal, never from the
training target. Each violation is attributed to one mechanism: ordinary mistake (M1 also
violates), cue-conditioned (it disappears when the cue line is removed), objective conflict
(targeted rule, not cue-dependent), or adapter drift (only untargeted rules broken).
`src/actmon/data/collect.py` defines these mechanism labels.

## Monitor access (the access table in every A2/A3 output, `access.csv`)

| Monitor | Activations | Labels | Extra actor passes per root | Weights | Context |
|---|---|---|---|---|---|
| response (proposed) | 3 sites, base + 6 transformed inputs | none | 6 | no | observable surface context |
| mahalanobis_pooled | 3 sites | none | 0 | no | none |
| mahalanobis_task | 3 sites | none | 0 | no | observable surface context |
| mahalanobis_env (invariance comparator) | 3 sites | none | 0 | no | template family |
| linear_probe | 3 sites | **dev violation labels** | 0 | no | none |
| contrast_probe | 3 sites | contrast instructions, no labels | 0 at test time | no | none |
| behavior_logistic | **none** | **dev violation labels** | 0 | no | observable surface context |
| strip_entropy | none (answers only) | none | 6 | no | none |
| teco_dispersion | none (answers only) | none | 6 | no | none |
| fa_style | none | none | 8 (scored) | **yes** | none |
| partial guard (reference row) | none | none | 0 | no | the *structured state*: it evaluates the reserve and handoff rules exactly |

No monitor receives the policy oracle. The task-changing transformations are structural edits
(toggle the authorization field; double or halve the amount). They never ask the oracle whether
the valid action changed. That property is computed afterwards, for diagnostics only. If a monitor
could query the oracle, it would not need activations, and the benchmark would be trivial. The
partial guard row shows how much a rule checker with access to the structured state already solves.

## Audit budget and FPR are different quantities

* The **FPR target** (1 %, 2 %, 5 %) is a conditional rate on normal roots, set on the calibration split.
* The **alarm fraction** is `TPR × prevalence + FPR × (1 − prevalence)`. It is reported at 10 %,
  1 % and 0.1 % prevalence, and as measured on the case-control test split.
* The **audit budget** is a resource allowance: the top 10, 20 or 50 of the 1,000 audit-cohort roots.

They are never used interchangeably.

## What the attacker knows and can do (A5)

* **Knows:** M2's weights, the attacked monitor's fitted parameters and its calibrated 1 %
  threshold, and the transformations the response detector applies. It has gradient access to all of them.
* **Controls:** `n_soft` continuous embedding vectors inserted at the end of the user message of
  *every* input the actor and the monitor process, including the six transformed inputs (the
  monitor edits the task text around them).
* **Cannot:** change the actor's weights, the monitor, the thresholds or the calibration data, or
  tell which roots will be audited.
* **Objective:** keep the violating proposal and push the monitor's score below its threshold, using
  one universal soft prompt trained on development violating roots and evaluated on test violating
  roots.

Covered: inference-time, white-box, continuous-embedding attacks on a frozen monitor. This is
stronger than any token-level suffix attacker.

Not covered: training-time attacks (an actor fine-tuned to obfuscate), discrete-token black-box
attackers, monitors retrained against the attack (the adversarial-training arms race of S16),
collusion between actor and monitor, and multi-step episodes where the actor reacts to blocks.

## Why this is not yet an AI-control evaluation

AI control (S01) evaluates a *protocol* (monitor, audit, fallback) against an *adaptive* adversary
with a stated budget, and measures safety and usefulness together. This study has:

* a controlled failure mechanism (the organism), which is **not** an adaptive adversary
* one white-box attack on the monitor (A5), which **is** adaptive but bounded to a soft prompt
* a utility measurement (A3) and a top-k audit (A3), but no measurement of executed failures in
  multi-step episodes

The next stage gives a bounded simulator attacker specified knowledge of the monitor and an
optimisation budget, then tests whether it can preserve a prohibited in-world objective while
evading detection across a full protocol. Until that runs, results are reported as "activation-based
monitoring". Nothing here claims adaptive-control or collusion resistance (Technical Specification,
"Future adaptive-control extension").
