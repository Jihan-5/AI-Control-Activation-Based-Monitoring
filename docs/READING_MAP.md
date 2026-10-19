# Reading map: syllabus → code (Research A, activation-based monitoring)

Each monitoring and shared-evaluation reading in the *Revised Core Reading Syllabus* (18 Oct 2026)
is mapped to the code that applies it, and to the artefact that answers its "done when" exercise.
The statistical-interpretability study (Research B) lives in its own repository. Override Lab readings (S03, S07, S08, S28–S31, T2
beyond loss masking) are not in this repository.

| ID | Reading | Applied in | "Done when" → artefact |
|---|---|---|---|
| S01 | Greenblatt et al., *AI Control* | `docs/THREAT_MODEL.md`; A3 (audit + fallback + utility); A5 (attack) | Defender and attacker access boundaries drawn; the study is named *activation-based monitoring* until an adaptive adversary evaluates the whole protocol; the falsifying attack is A5 |
| S04 | Miller, *Adding Error Bars to Evals* | `metrics.paired_tests` (i.i.d. and family-clustered paired differences); preregistration §4 power table | "55 % vs 62 % on 500" is insufficient: A2 reports the paired table, the clustered interval and the number of clusters (24 families) |
| S05 | Saito & Rehmsmeier, PR vs ROC | `metrics.detection_table` (precision and alarm fraction at 10 / 1 / 0.1 % prevalence); A3 audit | The same TPR/FPR gives different review precision at each prevalence; alarm fraction = TPR·π + FPR·(1−π) |
| S06 | Calibration assumptions | `evaluation/calibrate.py` (conformal threshold, exchangeability) | Achieved FPR next to every target; held-out families break exchangeability and are reported separately |
| S09 | Lee et al., Mahalanobis OOD | `detectors/mahalanobis.py` (pooled, task-conditioned) | Score implemented from scratch; every source of extra information listed in `access.csv`; no image-gradient preprocessing (no token analogue) |
| S10 | Goldowsky-Dill et al.; Anthropic sleeper-agent probes | `detectors/probes.py` (`linear_probe`, `contrast_probe`) | Matched probe at the same sites, thresholded on benign calibration data, paired comparison in A2; generic contrast-pair probe without knowledge of the planted violation |
| S13 | Carvalho et al., PCIR | `detectors/mahalanobis.EnvCentredMahalanobis` | Invariance comparator (family-mean removal); task conditioning ablated separately (`mahalanobis_task` vs `mahalanobis_env`); the transformation that would remove the violation signal is the TC group itself (A4 `response_tp_only`) |
| S14 | TeCo; STRIP-ViTA; ONION | `detectors/behavior.StripEntropy`, `TecoDispersion` | STRIP-style stability and TeCo-style severity dispersion with the same 6-query budget as the response detector; if they match it, the activation machinery adds nothing (A2 paired rows) |
| S15 | Lyons Keenan et al., functional attribution | `actor/functional.py`, `detectors/functional.py` (FA-style) | Access-and-cost table (weights, K scored passes); input-intervention responses vs parameter-perturbation traces are different objects (THREAT_MODEL access table) |
| S16 | Bailey et al., Obfuscated Activations | `attack/obfuscation.py`, A5 | White-box soft-prompt attack on the proposed monitor *and* both baselines; recall at fixed FPR before and after; violation kept (obfuscation tax); attacker model stated |
| T1 | ARENA; TransformerLens / nnsight | `actor/hooks.py` | Hook on `model.model.layers[i]` output verified against `hidden_states[i+1]`; left-padding invariance test; tokenizer fingerprint and resolved revision in `actor_meta.json` |
| T2 | TRL SFT loss masking; PEFT LoRA | `actor/train.py` (`masked_batch`), `actor/models.py` (LoRA, merge) | Assistant-only loss asserted in `test_assistant_only_loss_mask`; LoRA starts at the base model and merges exactly (`test_lora_starts_at_base_and_merges_exactly`) |
| X8 | Dietterich (exact McNemar) | `actmon.utils.stats.mcnemar_exact` | Exact small-sample paired test for every comparison |
