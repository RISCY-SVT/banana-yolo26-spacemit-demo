# H500 board decision

Decision: `fail`.

A1 EP - B2 EP mAP50-95: `0.006705286704`.
A1 EP/B2 EP prediction-count ratio: `0.980095322477`.

| Check | Status | Evidence |
|---|---|---|
| map_delta | pass | 0.006705286704 >= 0.005 |
| map_ci_lower | pass | 0.0014960370229444397 > 0 |
| ap_small_delta | pass | -0.003227026771 >= -0.005 |
| ap_medium_delta | pass | 0.005648627979 >= -0.005 |
| ap_large_delta | pass | 0.011143111601 >= -0.005 |
| ar_small_delta | fail | -0.009086253892 >= -0.005 |
| ar_medium_delta | pass | -0.001929875740 >= -0.005 |
| ar_large_delta | fail | -0.017721888749 >= -0.005 |
| a1_cpu_ep_map | fail | 0.001378162000 <= 0.001 |
| a1_cpu_ep_ap_small | pass | 0.001571861918 <= 0.003 |
| a1_cpu_ep_ap_medium | pass | 0.001642592641 <= 0.003 |
| a1_cpu_ep_ap_large | fail | 0.008548174752 <= 0.003 |
| a1_cpu_ep_ar_small | fail | 0.004332208775 <= 0.003 |
| a1_cpu_ep_ar_medium | pass | 0.001453427073 <= 0.003 |
| a1_cpu_ep_ar_large | fail | 0.015083694578 <= 0.003 |
| B2-cpu_failures | pass | 0 |
| B2-cpu_non_finite | pass | 0 |
| B2-spacemit_failures | pass | 0 |
| B2-spacemit_non_finite | pass | 0 |
| A1-cpu_failures | pass | 0 |
| A1-cpu_non_finite | pass | 0 |
| A1-spacemit_failures | pass | 0 |
| A1-spacemit_non_finite | pass | 0 |
