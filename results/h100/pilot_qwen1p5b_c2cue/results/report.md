# Run report: pilot_qwen1p5b_c2cue

Config digest `6c1f3bea5efaca60` · checkpoint `C2cue` · backend `hf`

## Primary comparison

```json
{
 "population": "familiar",
 "kind": "primary",
 "method": "proposed",
 "baseline": "mahalanobis_conditional",
 "tpr_method": 0.016,
 "tpr_baseline": 0.004,
 "tpr_diff": 0.012,
 "ci_low": 0.001968503937007874,
 "ci_high": 0.02385685884691849,
 "fpr_diff": 0.004,
 "fpr_diff_ci_low": -0.004960440606232562,
 "fpr_diff_ci_high": 0.013132310951459883,
 "method_only": 7,
 "baseline_only": 1,
 "mcnemar_p": 0.0703125,
 "n_clusters": 1500,
 "p_adjusted": 0.0703125
}
```
Decision: **no-go** — go/practically_useful: value=0.012 threshold=0.05; go/worst_group_fpr: value=0.04516129032258064 threshold=0.03

- [ablations](ablations.md)
- [access](access.md)
- [attack_curve](attack_curve.md)
- [control](control.md)
- [fpr_drift](fpr_drift.md)
- [operating_points](operating_points.md)
- [outcome_matched](outcome_matched.md)
- [paired](paired.md)
- [per_group_fpr](per_group_fpr.md)
- [tpr_by_mechanism](tpr_by_mechanism.md)
- [utility](utility.md)

## Gates

- screen: FAIL (2 checks)
