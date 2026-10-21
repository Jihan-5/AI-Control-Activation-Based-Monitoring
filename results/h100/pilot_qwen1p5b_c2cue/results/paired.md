### Paired comparisons at calibrated thresholds

> Measured on the configured backend; see run manifest for provenance.

Read tpr_diff together with fpr_diff: a method whose achieved FPR is inflated by shift also gains TPR.

| population | kind | method | baseline | tpr_method | tpr_baseline | tpr_diff | ci_low | ci_high | fpr_diff | fpr_diff_ci_low | fpr_diff_ci_high | method_only | baseline_only | mcnemar_p | n_clusters | p_adjusted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| familiar | primary | proposed | mahalanobis_conditional | 0.0160 | 0.0040 | 0.0120 | 0.0020 | 0.0239 | 0.0040 | -0.0050 | 0.0131 | 7 | 1 | 0.0703 | 1500 | 0.0703 |
| familiar | secondary | proposed | output_consistency | 0.0160 | 0.0220 | -0.0060 | -0.0167 | 0.0040 | 0.0090 | 0.0010 | 0.0174 | 2 | 5 | 0.4531 | 1500 | 0.4531 |
| familiar | secondary | proposed | mahalanobis_pooled | 0.0160 | 0.0020 | 0.0140 | 0.0036 | 0.0264 | 0.0040 | -0.0050 | 0.0130 | 8 | 1 | 0.0391 | 1500 | 0.1172 |
| familiar | secondary | proposed | behavior_logistic | 0.0160 | 0.3760 | -0.3600 | -0.4049 | -0.3150 | 0.0050 | -0.0050 | 0.0153 | 7 | 187 | 0.0000 | 1500 | 0.0000 |
| familiar | secondary | proposed | probe_supervised | 0.0160 | 0.0320 | -0.0160 | -0.0352 | 0.0021 | 0.0010 | -0.0101 | 0.0120 | 8 | 16 | 0.1516 | 1500 | 0.3032 |
| familiar | secondary@5% | proposed | mahalanobis_conditional | 0.0580 | 0.0260 | 0.0320 | 0.0101 | 0.0550 | 0.0140 | -0.0042 | 0.0322 | 25 | 9 | 0.0090 | 1500 | 0.0090 |
