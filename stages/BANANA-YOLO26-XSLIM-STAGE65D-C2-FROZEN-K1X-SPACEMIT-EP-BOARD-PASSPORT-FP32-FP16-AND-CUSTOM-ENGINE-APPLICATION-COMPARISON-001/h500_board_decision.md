# H500 board decision

Decision: `fail`.

C2 EP - B2 EP mAP50-95: `0.007548278134`.
C2 CPU/EP absolute mAP difference: `0.003144243371`.

| Check | Status | Evidence |
|---|---|---|
| c2_ep_b2_ep_map_delta | pass | 0.007548278134 >= 0.004 |
| c2_ep_b2_ep_map_probability | pass | 0.9996 >= 0.95 |
| ap_small_point | pass | -0.000863281913 >= -0.005 |
| ap_small_probability | pass | 0.9959 >= 0.90 |
| ap_medium_point | pass | 0.003130234316 >= -0.005 |
| ap_medium_probability | pass | 0.9991 >= 0.90 |
| ap_large_point | pass | 0.018154865179 >= -0.005 |
| ap_large_probability | pass | 1.0 >= 0.90 |
| ar_small_point | pass | -0.000052549998 >= -0.005 |
| ar_small_probability | pass | 0.9838 >= 0.90 |
| ar_medium_point | pass | -0.001679505083 >= -0.005 |
| ar_medium_probability | pass | 0.9015 >= 0.90 |
| ar_large_point | pass | -0.001609342455 >= -0.005 |
| ar_large_probability | pass | 0.9999 >= 0.90 |
| c2_cpu_ep_map_agreement | fail | 0.003144243371 <= 0.002 |
| c2_cpu_ep_ap_small | pass | 0.003023112862 <= 0.005 |
| c2_cpu_ep_ap_medium | pass | 0.003745156886 <= 0.005 |
| c2_cpu_ep_ap_large | fail | 0.010002015827 <= 0.005 |
| c2_cpu_ep_ar_small | pass | 0.003509889613 <= 0.005 |
| c2_cpu_ep_ar_medium | pass | 0.000386752907 <= 0.005 |
| c2_cpu_ep_ar_large | pass | 0.003682867807 <= 0.005 |
| B2_CPU_failures | pass | 0 |
| B2_CPU_non_finite | pass | 0 |
| B2_EP_failures | pass | 0 |
| B2_EP_non_finite | pass | 0 |
| C2_CPU_failures | pass | 0 |
| C2_CPU_non_finite | pass | 0 |
| C2_EP_failures | pass | 0 |
| C2_EP_non_finite | pass | 0 |

## Score-collapse census

All four surfaces have finite, nonconstant score distributions with multiple classes. Status: `pass`. See `h500_score_distribution.tsv`.
