# Stage 11 Branch Block Oracle Report

model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
selected_subset: `candidate_F_model2_m0_cv1_act_cv2_conv`
output_boundary: corrected int32 output of `/model.2/m.0/cv2/conv/Conv`

## New Boundary

- producer: `/model.2/m.0/cv1/conv/Conv`
- activation: `/model.2/m.0/cv1/act/Sigmoid` + `/model.2/m.0/cv1/act/Mul`
- consumer: `/model.2/m.0/cv2/conv/Conv`
- conv_output_scale: `0.038180503994226456`
- conv_output_zero_point_u8: `176`
- act_output_scale: `0.012377118691802025`
- act_output_zero_point_u8: `22`
- act_input_storage_zero_point_s8: `-106`

## Boundary 256-Code LUT Oracle

- onnx_lut_model: `/data/ncnn-logs/ai-team/2026-07-04_19-42-26/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE11-BRANCH-BLOCK-EXPANSION-001/run_logs/stage11_branch0_to_act_lut.onnx`
- onnx_lut_model_sha256: `8af7be0904e1e61603d82b59bffd9dab421f122cc80fc024a8242eb955f6ebc7`
- mismatches: `0`
- max_abs_diff_u8: `0`

## Branch Conv2

- node: `/model.2/m.0/cv2/conv/Conv`
- input_shape_nhwc: `[2, 2, 8]` for compact fixture
- output_shape_nhwc: `[2, 2, 16]` for compact fixture
- kernel: `3x3`
- stride: `1x1`
- padding: `1`
- output_scale: `0.5276883244514465`
- output_zero_point_u8: `179`
- weight_scale_count: `16`
- weight_zero_points: all `0`

## Small Fixture Checksums

- seeded_branch0_act_sum: `-3384`
- seeded_branch1_i32_sum: `5006`
- gradient_branch0_act_sum: `-3057`
- gradient_branch1_i32_sum: `9654`

## Residual Add

Residual Add is not included in the generated Stage 11A fixture. ONNX represents
`/model.2/m.0/Add` as a float-domain Add of
`/model.2/Split_output_1_DequantizeLinear_Output` and
`/model.2/m.0/cv2/act/Mul_output_0`. There is no clean integer Add output
contract before the later Concat Q/DQ boundary in this stage.
