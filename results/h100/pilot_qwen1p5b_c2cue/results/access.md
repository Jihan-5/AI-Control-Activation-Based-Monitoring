### Access table

> Measured on the configured backend; see run manifest for provenance.

| method | activations | action_scores | forwards_per_root | weights | backward_passes | violation_labels | normal_only | oracle_rules | notes |
|---|---|---|---|---|---|---|---|---|---|
| proposed | True | True | 7.0000 | False | False | False | True | False | normal-fit only; 6 edited inputs per root plus candidate scoring |
| mahalanobis_pooled | True | False | 1.0000 | False | False | False | True | False | single base forward; normal-fit only |
| mahalanobis_conditional | True | False | 1.0000 | False | False | False | True | False | single base forward; conditions on observable context; normal-fit only |
| output_consistency | False | True | 7.0000 | False | False | False | True | False | no activations; same edited inputs as the proposed detector |
| teco | False | True | 17.0000 | False | False | False | True | False | TeCo-style adaptation: 4 edit kinds x 4 severities |
| strip | False | True | 9.0000 | False | False | False | True | False | STRIP-style adaptation: 8 perturbations |
| pcir | True | False | 1.0000 | False | False | False | True | False | single base forward; uses observable environment labels; normal-fit only; adaptation |
| functional_attribution | False | False | 17.0000 | True | False | False | True | False | 16 parameter-sampling forwards per root; class-conditioned references; adaptation |
| probe_supervised | True | False | 1.0000 | False | False | True | False | False | trained on labelled dev data: extra-information reference, not equal-data baseline |
| probe_contrast | True | False | 1.0000 | False | False | False | True | False | generic contrast direction from contrast prompts; no violation labels |
| behavior_logistic | False | True | 1.0000 | False | False | True | False | False | no activations; trained on labelled dev data (disclosed label access) |
| oracle | False | True | 1.0000 | False | False | False | True | True | exact policy rules: ceiling, not a deployable monitor in general |
| metadata_shortcut | False | False | 0.0000 | False | False | True | False | False | leakage diagnostic on non-semantic metadata; never a monitor |
