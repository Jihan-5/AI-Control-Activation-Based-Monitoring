"""irmon: Intervention-Response MONitoring under benign distribution shift.

Research code for the study "When Unusual Is Not Unsafe: Intervention-Response
Monitoring under Benign Distribution Shift".

STATUS: research infrastructure only. No language-model experiment has been run
with this code yet, and nothing in this package should be read as a measured
result. Synthetic backends exist to test the pipeline, not to estimate effects.

Package map
-----------
envs/           Task environments with exact rule oracles (finite-state decisions,
                bounded program reasoning).
interventions/  T (nuisance, decision-preserving) and U (task-changing) edit
                operators with programmatic validation.
data/           Split design, immutable manifests, leakage guards and audits.
models/         Actor backends (Hugging Face + synthetic), residual hooks,
                candidate-set action scoring, LoRA organism training.
extraction/     Activation/score extraction and on-disk caches.
features/       Response bundles, training-only projections, signatures.
detectors/      The proposed intervention-response detector and every baseline.
calibration/    Split-conformal thresholds and per-group FPR auditing.
evaluation/     Metrics, paired/clustered statistics, control protocols, gates.
theory/         Restricted theoretical model (nuisance subspace vs failure shift).
attacks/        Detector-aware stress tests (black-box search, soft prompts).
pipeline/       Stage orchestration with go/no-go gates.
"""

__version__ = "0.1.0"

EVIDENCE_STATUS = (
    "No model experiment has been executed with this repository. All numbers in "
    "docs are design choices or illustrative arithmetic, not findings."
)
