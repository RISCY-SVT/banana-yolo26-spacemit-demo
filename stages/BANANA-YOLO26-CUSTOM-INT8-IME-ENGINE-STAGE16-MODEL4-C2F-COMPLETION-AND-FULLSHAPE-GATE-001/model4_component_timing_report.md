# Model4 Component Timing Report

subset: `candidate_J_model4_c2f_complete_compact`
shape_class: `compact`

| candidate | total_us | conv_us | activation_requant_us | split_us | merge_us | add_us | concat_us | post_qdq_us | pack_layout_us | correction_us | branch1_conv_us | branch1_activation_us | model4_cv2_conv_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | 418.612 | 287.87 | 88.573 | 9.76433 | 24.084 | 10.917 | 10.917 | 14.3197 | 0.375 | 4.17967 | 11.6393 | 4.31967 | 21.2643 | 68.7678 | 21.1587 | 5.7533 | 0.0895818 |
| `stage16_IME_A2_rvv_f32_lut` | 197.493 | 128.992 | 29.098 | 9.33367 | 23.001 | 10.7643 | 10.7643 | 13.6673 | 0.416667 | 3.416 | 7.23633 | 4.236 | 15.042 | 65.3146 | 14.7337 | 11.6465 | 0.210978 |

This compact timing is correctness/local proportion evidence only. The Stage16A representative/full-shape gate must be used for performance decisions.
