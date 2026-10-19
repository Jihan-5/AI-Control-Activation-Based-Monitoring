| detector | supervised | activations | labels | extra_forward_passes | weights | context | notes | fit_sec |
|---|---|---|---|---|---|---|---|---|
| response | False | sites+variants | none | 6 | False | observable surface context | groups=tp+tc; context=base+surface; structural edits, no oracle | 0.892 |
| mahalanobis_pooled | False | sites | none | 0 | False | none | normal fitting split only | 0.041 |
| mahalanobis_task | False | sites | none | 0 | False | observable surface context | class = resource type x handoff message present x pool count | 0.244 |
| mahalanobis_env | False | sites | none | 0 | False | template family | invariance comparator: family-mean removal | 0.088 |
| linear_probe | True | sites | dev violation labels | 0 | False | none | fit on dev half A, C tuned on dev half B | 0.309 |
| contrast_probe | False | sites | contrast instructions (no labels) | 0 | False | none | 2 extra passes per fitting root only; none at test time | 0.090 |
| behavior_logistic | True | none | dev violation labels | 0 | False | observable surface context | action scores + verb + surface context; fit dev half A, tuned on dev half B | 0.043 |
| strip_entropy | False | none | none | 6 | False | none | actor answers under the six main-suite transforms | 0.000 |
| teco_dispersion | False | none | none | 6 | False | none | hard labels under stacked rename/reorder/paraphrase, severity 1-2 | 0.000 |
| fa_style | False | none | none | 8 | True | none | K parameter-perturbed scored passes per root | 0.001 |
