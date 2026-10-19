# Reading Map: syllabus → code

How each reading from the *Revised Core Reading Syllabus* (18 Oct 2026) that serves the monitoring
workstream is used in this repository. IDs follow the syllabus. "Done when" checks are the syllabus's
comprehension gates; the right column says which artifact demonstrates them.

| ID | Reading | Depth | Where it shows up in code / docs | "Done when" artifact |
|---|---|---|---|---|
| S01 | Greenblatt et al., *AI Control* (ICML 2024) | Deep | `docs/THREAT_MODEL.md`; naming discipline (activation-based monitoring until an adaptive adversary is evaluated); `evaluation/control.py` audit-budget protocols | Threat model with access boundaries and a falsifying attack |
| P1 | Wasserman, *All of Statistics* ch. 5, 7–10 | Prereq | `evaluation/stats.py` (bootstrap, delta-method-free percentile CIs, multiple testing) | Paired cluster bootstrap + Holm in `tests/test_stats.py` |
| S04 | Miller, *Adding Error Bars to Evals* | Targeted | `evaluation/stats.py::clustered_paired_difference`, `paired_cluster_bootstrap`; power helpers | Clustered paired interval from root-level logs; cluster counts reported |
| S05 | Saito & Rehmsmeier, PR vs ROC | Targeted | `evaluation/metrics.py` (AUPRC, PPV at prevalence, alarm fraction) | PPV at 10%/1%/0.1% for one TPR/FPR pair (test) |
| S06 | Angelopoulos & Bates, conformal tutorial | Targeted | `calibration/conformal.py` (split conformal, Beta coverage law, per-group FPR) | What is random/fixed and which guarantee fails under shift (docstrings) |
| T1 | ARENA ch.1, TransformerLens / nnsight, SAELens | Practice | `models/hf_backend.py` (plain HF forward hooks, no weight processing), `verify_hooks` | Hook capture reproduces a manual forward to tolerance; library versions in manifest |
| T2 | TRL SFT + PEFT LoRA docs | Practice | `models/lora.py` (custom loop, assistant-only masking, base-weight assertion) | Saved masked-example printout; test fails if a base weight changes |
| S07 | Hu et al., *LoRA* | Targeted | `models/lora.py`, `configs/lora_*.yaml`, adapter lineage JSON | Audit which weights changed; reproduce starting checkpoint |
| S08 | Biderman et al., *LoRA Learns Less and Forgets Less* | Targeted | E3: q/v-only vs predeclared all-linear adapter on dev; learn–forget tracking | Dev comparison of two adapter configs |
| S09 | Lee et al., Mahalanobis OOD | Targeted | `detectors/mahalanobis.py` (pooled + task-conditioned; no image preprocessing transplanted) | Score implemented from scratch; extra-information sources listed in access table |
| S10 | Goldowsky-Dill et al.; Anthropic sleeper-agent probes | Deep | `detectors/probes.py` (supervised probe, contrast-pair probe), access disclosure | Advantage over matched probe at fixed FPR with paired interval, or a narrowed claim |
| S13 | Carvalho et al., PCIR | Deep | `detectors/pcir.py`; *no-conditioning* ablation separate from invariance | Case where removing environmental variation also removes the violation signal (T1, E8) |
| S14 | TeCo; STRIP-ViTA; ONION | Targeted | `detectors/teco.py`, `detectors/strip.py`, `interventions/severity.py` | Baselines that could disprove the method's addition at matched query budget |
| S15 | Lyons Keenan et al., functional-attribution MAD | Deep | `detectors/functional_attribution.py`, `HFBackend.loss_trace` | Access-and-cost table row; cost–quality curve rather than an underpowered config |
| S16 | Bailey et al., *Obfuscated Activations* | Deep | `attacks/black_box.py`, `attacks/soft_prompt.py`; E12 | Attack curve for the monitor and both baselines; attacker model stated |
| S03 | Anthropic, *Agentic Misalignment* | Deep | Organism framing in `envs/finite_state/planner.py`; claim discipline in docs | Distinguish organism vs elicitation vs training intervention vs risk estimate |
| S28 | Emergent Misalignment (conditional for this repo) | — | Read before training C2 (syllabus hard constraint) | — |
| X8 | Dietterich, paired tests | Conditional | `evaluation/stats.py::mcnemar_exact` | Exact small-sample paired test |

Hard ordering constraints honoured: S16 before any robustness claim; S08 and S28 before training the
objective-biased organism; the theory gate (T1 simulation) before collecting language-model activations.

Readings for the statistical-interpretability (S02, S11, S12, S17–S27) and Override Lab (S29–S31)
workstreams are out of scope for this repository.
