# Component Timing Report

selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`
board_cpu: `CPU0`
path: `stage12_IME_A2_rvv_f32_lut`

| bucket | mean_us |
|---|---:|
| `conv_us` | `272934` |
| `activation_requant_us` | `87946.4` |
| `split_us` | `128991` |
| `add_us` | `2504.42` |
| `concat_us` | `4335.56` |
| `post_concat_qdq_us` | `83007.7` |
| `pack_layout_us` | `129778` |
| `correction_us` | `3439.52` |
| `model2_cv2_conv_us` | `50056.6` |
| `total_us` | `582039` |

Shares:

- `activation_share_pct=15.1699`
- `conv_share_pct=47.0785`
- `add_concat_share_pct=15.4979`
- `pack_layout_share_pct=22.3855`

Interpretation: the C2f boundary is correct, but the measured float split/materialization
and post-Concat QDQ path is a visible bottleneck. This is a local selected-subset
finding, not a full-model performance claim.
