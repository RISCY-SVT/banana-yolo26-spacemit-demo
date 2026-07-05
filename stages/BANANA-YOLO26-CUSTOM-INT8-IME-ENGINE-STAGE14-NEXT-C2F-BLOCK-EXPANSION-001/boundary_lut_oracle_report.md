# Stage 14 Boundary LUT Oracle Report

model: `/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`

## Boundaries

| boundary | conv_scale | conv_zp | act_scale | act_zp | ort_mismatches | max_abs_diff_u8 | micro_model |
|---|---:|---:|---:|---:|---:|---:|---|
| `/model.2/cv2/act` | `0.4553883671760559` | `186` | `0.12448897212743759` | `2` | `0` | `0` | `.deps/custom_int8_engine/stage14_oracle/stage14_model2_cv2_activation.onnx` |
| `/model.3/act` | `0.10164494067430496` | `157` | `0.04000610113143921` | `7` | `0` | `0` | `.deps/custom_int8_engine/stage14_oracle/stage14_model3_activation.onnx` |

## Decision

Both Stage 14 activation boundaries use boundary-specific 256-code ONNX Runtime
micro-oracles. Accepted path requires `ort_mismatches=0`.
