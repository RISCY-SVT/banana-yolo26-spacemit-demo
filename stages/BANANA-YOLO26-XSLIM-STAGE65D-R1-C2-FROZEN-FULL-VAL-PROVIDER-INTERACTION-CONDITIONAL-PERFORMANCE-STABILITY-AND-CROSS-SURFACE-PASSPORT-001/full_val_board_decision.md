# Stage65D-R1 full val2017 decision

Primary C2 EP versus B2 EP task gate: `fail`.
C2 EP - B2 EP mAP50-95: `0.012599140370`; 95% CI `[0.011641520920459936, 0.015099894258865584]`.
Provider interaction material metrics: `0`. Absolute CPU/EP point gaps are warnings only in this Stage.

| Check | Status | Evidence |
|---|---|---|
| c2_ep_b2_ep_map_delta | pass | 0.012599140370 >= 0.005 |
| c2_ep_b2_ep_map_ci | pass | 0.011641520920459936 > 0 |
| ap_small_point | pass | -0.000253861755 >= -0.003 |
| ap_small_ci | pass | -0.001394469868066902 >= -0.005 |
| ap_medium_point | pass | 0.006319481209 >= -0.003 |
| ap_medium_ci | pass | 0.005236853724387781 >= -0.005 |
| ap_large_point | pass | 0.039947408575 >= -0.003 |
| ap_large_ci | pass | 0.030634456921003823 >= -0.005 |
| ar_small_point | fail | -0.003444784993 >= -0.003 |
| ar_small_ci | fail | -0.006213496738686553 >= -0.005 |
| ar_medium_point | pass | -0.001042974506 >= -0.003 |
| ar_medium_ci | pass | -0.002046535767054211 >= -0.005 |
| ar_large_point | fail | -0.003078277927 >= -0.003 |
| ar_large_ci | pass | -0.00469129207818266 >= -0.005 |
| B2_CPU_images | pass | 5000 |
| B2_CPU_failures | pass | 0 |
| B2_CPU_non_finite | pass | 0 |
| B2_CPU_score_collapse | pass | False |
| B2_EP_images | pass | 5000 |
| B2_EP_failures | pass | 0 |
| B2_EP_non_finite | pass | 0 |
| B2_EP_score_collapse | pass | False |
| C2_CPU_images | pass | 5000 |
| C2_CPU_failures | pass | 0 |
| C2_CPU_non_finite | pass | 0 |
| C2_CPU_score_collapse | pass | False |
| C2_EP_images | pass | 5000 |
| C2_EP_failures | pass | 0 |
| C2_EP_non_finite | pass | 0 |
| C2_EP_score_collapse | pass | False |
