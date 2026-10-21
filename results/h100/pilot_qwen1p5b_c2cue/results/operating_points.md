### Locked-test operating points (calibrated thresholds)

> Measured on the configured backend; see run manifest for provenance.

| method | nominal_fpr | threshold | tpr | achieved_fpr | auroc | auprc_benchmark | tp | fp | fn | tn | worst_group | worst_group_fpr | ppv@0.01 | alarms@0.01 | ppv@0.001 | alarms@0.001 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| behavior_logistic | 0.0100 | 3.3148 | 0.3120 | 0.0110 | 0.9607 | 0.9163 | 312 | 22 | 688 | 1978 | unfamiliar|no_cue|no_conflict_prompt | 0.0195 | 0.2227 | 0.0140 | 0.0276 | 0.0113 |
| behavior_logistic | 0.0200 | 2.8722 | 0.4190 | 0.0145 | 0.9607 | 0.9163 | 419 | 29 | 581 | 1971 | familiar|no_cue|conflict_prompt | 0.0221 | 0.2259 | 0.0185 | 0.0281 | 0.0149 |
| behavior_logistic | 0.0500 | 1.3151 | 0.8090 | 0.0395 | 0.9607 | 0.9163 | 809 | 79 | 191 | 1921 | unfamiliar|no_cue|no_conflict_prompt | 0.0488 | 0.1714 | 0.0472 | 0.0201 | 0.0403 |
| functional_attribution | 0.0100 | 1.4726 | 0.0110 | 0.0100 | 0.5433 | 0.3623 | 11 | 20 | 989 | 1980 | unfamiliar|cue_present|no_conflict_prompt | 0.0190 | 0.0110 | 0.0100 | 0.0011 | 0.0100 |
| functional_attribution | 0.0200 | 1.3979 | 0.0270 | 0.0210 | 0.5433 | 0.3623 | 27 | 42 | 973 | 1958 | unfamiliar|cue_present|no_conflict_prompt | 0.0333 | 0.0128 | 0.0211 | 0.0013 | 0.0210 |
| functional_attribution | 0.0500 | 1.2935 | 0.0580 | 0.0470 | 0.5433 | 0.3623 | 58 | 94 | 942 | 1906 | unfamiliar|no_cue|conflict_prompt | 0.0670 | 0.0123 | 0.0471 | 0.0012 | 0.0470 |
| mahalanobis_conditional | 0.0100 | 256.0359 | 0.0060 | 0.0170 | 0.5025 | 0.3246 | 6 | 34 | 994 | 1966 | unfamiliar|cue_present|no_conflict_prompt | 0.0381 | 0.0036 | 0.0169 | 0.0004 | 0.0170 |
| mahalanobis_conditional | 0.0200 | 233.9591 | 0.0140 | 0.0275 | 0.5025 | 0.3246 | 14 | 55 | 986 | 1945 | unfamiliar|cue_present|no_conflict_prompt | 0.0571 | 0.0051 | 0.0274 | 0.0005 | 0.0275 |
| mahalanobis_conditional | 0.0500 | 206.9005 | 0.0480 | 0.0640 | 0.5025 | 0.3246 | 48 | 128 | 952 | 1872 | familiar|cue_present|conflict_prompt | 0.1000 | 0.0075 | 0.0638 | 0.0008 | 0.0640 |
| mahalanobis_pooled | 0.0100 | 249.7657 | 0.0040 | 0.0155 | 0.4893 | 0.3172 | 4 | 31 | 996 | 1969 | unfamiliar|cue_present|no_conflict_prompt | 0.0286 | 0.0026 | 0.0154 | 0.0003 | 0.0155 |
| mahalanobis_pooled | 0.0200 | 226.7728 | 0.0140 | 0.0270 | 0.4893 | 0.3172 | 14 | 54 | 986 | 1946 | unfamiliar|cue_present|no_conflict_prompt | 0.0476 | 0.0052 | 0.0269 | 0.0005 | 0.0270 |
| mahalanobis_pooled | 0.0500 | 194.8256 | 0.0560 | 0.0750 | 0.4893 | 0.3172 | 56 | 150 | 944 | 1850 | unfamiliar|cue_present|no_conflict_prompt | 0.1048 | 0.0075 | 0.0748 | 0.0007 | 0.0750 |
| metadata_shortcut | 0.0100 | 0.5668 | 0.0050 | 0.0050 | 0.5275 | 0.3497 | 5 | 10 | 995 | 1990 | familiar|no_cue|conflict_prompt | 0.0133 | 0.0100 | 0.0050 | 0.0010 | 0.0050 |
| metadata_shortcut | 0.0200 | 0.5146 | 0.0140 | 0.0100 | 0.5275 | 0.3497 | 14 | 20 | 986 | 1980 | familiar|cue_present|conflict_prompt | 0.0250 | 0.0139 | 0.0100 | 0.0014 | 0.0100 |
| metadata_shortcut | 0.0500 | 0.4148 | 0.0460 | 0.0380 | 0.5275 | 0.3497 | 46 | 76 | 954 | 1924 | familiar|no_cue|no_conflict_prompt | 0.0505 | 0.0121 | 0.0381 | 0.0012 | 0.0380 |
| oracle | 0.0100 | 1.0000 | 0.0000 | 0.0000 | 0.9822 | 0.9578 | 0 | 0 | 1000 | 2000 | familiar|no_cue|no_conflict_prompt | 0.0000 | nan | 0.0000 | nan | 0.0000 |
| oracle | 0.0200 | 0.0000 | 0.9790 | 0.0145 | 0.9822 | 0.9578 | 979 | 29 | 21 | 1971 | familiar|cue_present|no_conflict_prompt | 0.0302 | 0.4055 | 0.0241 | 0.0633 | 0.0155 |
| oracle | 0.0500 | 0.0000 | 0.9790 | 0.0145 | 0.9822 | 0.9578 | 979 | 29 | 21 | 1971 | familiar|cue_present|no_conflict_prompt | 0.0302 | 0.4055 | 0.0241 | 0.0633 | 0.0155 |
| output_consistency | 0.0100 | 281.3043 | 0.0240 | 0.0090 | 0.5562 | 0.3748 | 24 | 18 | 976 | 1982 | unfamiliar|cue_present|no_conflict_prompt | 0.0190 | 0.0262 | 0.0092 | 0.0027 | 0.0090 |
| output_consistency | 0.0200 | 198.0131 | 0.0560 | 0.0385 | 0.5562 | 0.3748 | 56 | 77 | 944 | 1923 | unfamiliar|cue_present|no_conflict_prompt | 0.0762 | 0.0145 | 0.0387 | 0.0015 | 0.0385 |
| output_consistency | 0.0500 | 149.6072 | 0.0980 | 0.0810 | 0.5562 | 0.3748 | 98 | 162 | 902 | 1838 | unfamiliar|no_cue|conflict_prompt | 0.1443 | 0.0121 | 0.0812 | 0.0012 | 0.0810 |
| pcir | 0.0100 | 4.6472 | 0.0030 | 0.0070 | 0.5069 | 0.3300 | 3 | 14 | 997 | 1986 | familiar|cue_present|conflict_prompt | 0.0250 | 0.0043 | 0.0070 | 0.0004 | 0.0070 |
| pcir | 0.0200 | 3.9304 | 0.0070 | 0.0125 | 0.5069 | 0.3300 | 7 | 25 | 993 | 1975 | unfamiliar|cue_present|conflict_prompt | 0.0476 | 0.0056 | 0.0124 | 0.0006 | 0.0125 |
| pcir | 0.0500 | 2.7251 | 0.0250 | 0.0395 | 0.5069 | 0.3300 | 25 | 79 | 975 | 1921 | familiar|cue_present|conflict_prompt | 0.1625 | 0.0064 | 0.0394 | 0.0006 | 0.0395 |
| probe_contrast | 0.0100 | 3.9492 | 0.0070 | 0.0070 | 0.5062 | 0.3440 | 7 | 14 | 993 | 1986 | familiar|no_cue|no_conflict_prompt | 0.0202 | 0.0100 | 0.0070 | 0.0010 | 0.0070 |
| probe_contrast | 0.0200 | 3.5388 | 0.0150 | 0.0140 | 0.5062 | 0.3440 | 15 | 28 | 985 | 1972 | familiar|no_cue|no_conflict_prompt | 0.0343 | 0.0107 | 0.0140 | 0.0011 | 0.0140 |
| probe_contrast | 0.0500 | 3.0068 | 0.0380 | 0.0360 | 0.5062 | 0.3440 | 38 | 72 | 962 | 1928 | familiar|no_cue|no_conflict_prompt | 0.0606 | 0.0105 | 0.0360 | 0.0011 | 0.0360 |
| probe_supervised | 0.0100 | 2.2289 | 0.0200 | 0.0090 | 0.5992 | 0.4070 | 20 | 18 | 980 | 1982 | familiar|no_cue|conflict_prompt | 0.0398 | 0.0220 | 0.0091 | 0.0022 | 0.0090 |
| probe_supervised | 0.0200 | 2.0043 | 0.0370 | 0.0165 | 0.5992 | 0.4070 | 37 | 33 | 963 | 1967 | familiar|no_cue|conflict_prompt | 0.0708 | 0.0221 | 0.0167 | 0.0022 | 0.0165 |
| probe_supervised | 0.0500 | 1.7179 | 0.0690 | 0.0360 | 0.5992 | 0.4070 | 69 | 72 | 931 | 1928 | familiar|no_cue|conflict_prompt | 0.1150 | 0.0190 | 0.0363 | 0.0019 | 0.0360 |
| proposed | 0.0100 | 2254.1284 | 0.0110 | 0.0125 | 0.5310 | 0.3513 | 11 | 25 | 989 | 1975 | familiar|no_cue|no_conflict_prompt | 0.0222 | 0.0088 | 0.0125 | 0.0009 | 0.0125 |
| proposed | 0.0200 | 2043.8849 | 0.0240 | 0.0260 | 0.5310 | 0.3513 | 24 | 52 | 976 | 1948 | familiar|no_cue|no_conflict_prompt | 0.0485 | 0.0092 | 0.0260 | 0.0009 | 0.0260 |
| proposed | 0.0500 | 1836.0916 | 0.0630 | 0.0535 | 0.5310 | 0.3513 | 63 | 107 | 937 | 1893 | familiar|no_cue|no_conflict_prompt | 0.0808 | 0.0118 | 0.0536 | 0.0012 | 0.0535 |
| strip | 0.0100 | 3.1718 | 0.0020 | 0.0105 | 0.4993 | 0.3294 | 2 | 21 | 998 | 1979 | familiar|no_cue|conflict_prompt | 0.0177 | 0.0019 | 0.0104 | 0.0002 | 0.0105 |
| strip | 0.0200 | 2.5864 | 0.0180 | 0.0260 | 0.4993 | 0.3294 | 18 | 52 | 982 | 1948 | unfamiliar|no_cue|no_conflict_prompt | 0.0371 | 0.0069 | 0.0259 | 0.0007 | 0.0260 |
| strip | 0.0500 | 1.9295 | 0.0720 | 0.0670 | 0.4993 | 0.3294 | 72 | 134 | 928 | 1866 | unfamiliar|no_cue|no_conflict_prompt | 0.1055 | 0.0107 | 0.0670 | 0.0011 | 0.0670 |
| teco | 0.0100 | 2.0000 | 0.0000 | 0.0000 | 0.5485 | 0.3571 | 0 | 0 | 1000 | 2000 | familiar|no_cue|no_conflict_prompt | 0.0000 | nan | 0.0000 | nan | 0.0000 |
| teco | 0.0200 | 1.7854 | 0.0190 | 0.0190 | 0.5485 | 0.3571 | 19 | 38 | 981 | 1962 | familiar|no_cue|conflict_prompt | 0.0310 | 0.0100 | 0.0190 | 0.0010 | 0.0190 |
| teco | 0.0500 | 1.7854 | 0.0190 | 0.0190 | 0.5485 | 0.3571 | 19 | 38 | 981 | 1962 | familiar|no_cue|conflict_prompt | 0.0310 | 0.0100 | 0.0190 | 0.0010 | 0.0190 |
