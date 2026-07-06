# Component Timing Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

## Stable Board Protocol

```text
pinning: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
```

## Representative/Full-Shape C2 Transfer

| bucket | C2_split0_concat_lut_4t mean_us |
|---|---:|
| total_us | 116631 |
| conv_us | 53165.3 |
| activation_requant_us | 29787.3 |
| merge_us | 29767 |
| thread_overhead_us | 576.065 |
| branch0_conv_us | 6164.37 |
| branch1_conv_us | 15884.2 |
| model4_cv2_conv_us | 31116.7 |
| correction_us | 2739.91 |
| conv_share_pct | 45.5842 |
| activation_share_pct | 25.5398 |
| merge_share_pct | 25.5224 |

## Real Runner Integrated Compact C2

| bucket | C2_integrated_model4_c2f mean_us |
|---|---:|
| total_us | 284.289158 |
| conv_us | 218.501106 |
| activation_requant_us | 31.662926 |
| merge_us | 17.561526 |
| branch0_conv_us | 86.543282 |
| branch1_conv_us | 21.945104 |
| model4_cv2_conv_us | 15.432106 |
| add_us | 5.081710 |
| concat_us | 5.081710 |
| post_concat_qdq_us | 7.943548 |
| thread_overhead_us | 77.483640 |
| conv_share_pct | 76.858754 |
| activation_share_pct | 11.137578 |
| merge_share_pct | 6.177346 |

Compact timing is local correctness/proportional evidence only.
