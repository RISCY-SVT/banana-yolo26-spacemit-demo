# Stage 14 Block Oracle Report

model: `/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
selected_subset: `candidate_H3_model2_act_model3_act_model4_cv1_conv`
output_boundary: corrected int32 output of `/model.4/cv1/conv/Conv`

## Included Nodes

1. Stage 13 selected subset through corrected int32 output of `/model.2/cv2/conv/Conv`
2. `/model.2/cv2/act/Sigmoid`
3. `/model.2/cv2/act/Mul`
4. `/model.2/cv2/act/Mul_output_0` Q/DQ handoff
5. `/model.3/conv/Conv`
6. `/model.3/act/Sigmoid`
7. `/model.3/act/Mul`
8. `/model.3/act/Mul_output_0` Q/DQ handoff
9. `/model.4/cv1/conv/Conv`

The subset stops before `/model.4/Split`.

## Conv Metadata

| node | compact input NHWC | compact output NHWC | kernel | stride | pad | output_scale | output_zp | weight_shape_oihw | weight_zp_all_zero |
|---|---|---|---|---|---|---:|---:|---|---|
| `/model.3/conv/Conv` | `[2,2,64]` | `[1,1,64]` | `3x3` | `2` | `1` | `0.10164494067430496` | `157` | `[64, 64, 3, 3]` | `True` |
| `/model.4/cv1/conv/Conv` | `[1,1,64]` | `[1,1,64]` | `1x1` | `1` | `0` | `0.06883347779512405` | `163` | `[64, 64, 1, 1]` | `True` |

## Compact Fixture Checksums

| fixture | model3_input_s8_sum | model3_i32_sum | model4_cv1_input_s8_sum | model4_cv1_i32_sum |
|---|---:|---:|---:|---:|
| synthetic_seeded | `-30060` | `471248` | `-6766` | `340262` |
| synthetic_gradient | `-30125` | `640493` | `-6614` | `373793` |

## Boundary LUT Oracle

- `/model.2/cv2/act` mismatches: `0`, max_abs_diff_u8: `0`
- `/model.3/act` mismatches: `0`, max_abs_diff_u8: `0`

## Decision

Stage 14 selects Candidate H3 because it expands past `/model.2` to `/model.3`
and `/model.4/cv1` without crossing the next `/model.4/Split` branch point.
