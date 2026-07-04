# Component Timing Report

scope: `candidate_F_model2_m0_cv1_act_cv2_conv`
board: `svt@banana`
affinity: CPU0

| component | Stage11 IME A2 us |
|---|---:|
| conv0_ime_us | 68437.1 |
| act0_requant_lut_us | 19973.5 |
| conv1_ime_us | 63761.7 |
| act1_requant_lut_us | 7496.89 |
| conv2_ime_us | 25868.8 |
| act2_requant_lut_us | 8577.45 |
| split_copy_us | 1146.1 |
| branch_cv1_conv_us | 39434.4 |
| branch_cv1_activation_us | 4022.68 |
| branch_cv2_conv_us | 30063.5 |
| branch_cv2_correction_us | 1017.85 |
| residual_add_us | 0 |
| concat_copy_us | 0 |
| total_us | 269372 |

Shares:

- activation_share: `14.8755%`
- conv_share: `84.4801%`
- pack_layout_share: `0.425471%`

The dominant bucket remains Conv/IME compute for this selected subset.
