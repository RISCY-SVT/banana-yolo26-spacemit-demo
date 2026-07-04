# Stage 12 C2f Oracle Report

model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`
output_boundary: corrected int32 output of `/model.2/cv2/conv/Conv`

## Float-Domain Merge Contract

- Add node: `/model.2/m.0/Add`
- Add inputs:
  - `/model.2/Split_output_1_DequantizeLinear_Output`
  - `/model.2/m.0/cv2/act/Mul_output_0`
- Add output: `/model.2/m.0/Add_output_0`
- Concat node: `/model.2/Concat`
- Concat inputs:
  - `/model.2/Split_output_0`
  - `/model.2/Split_output_1_DequantizeLinear_Output`
  - `/model.2/m.0/Add_output_0`
- Concat output: `/model.2/Concat_output_0`
- post-Concat Q/DQ scale: `0.3288085460662842`
- post-Concat Q/DQ zero_point_u8: `2`

## Model2 Cv2 Conv

- node: `/model.2/cv2/conv/Conv`
- compact input_shape_nhwc: `[2, 2, 48]`
- compact output_shape_nhwc: `[2, 2, 64]`
- kernel: `1x1`
- stride: `1x1`
- padding: `0`
- output_scale: `0.4553883671760559`
- output_zero_point_u8: `186`
- weight_shape_oihw: `[64, 48, 1, 1]`
- weight_scale_count: `64`
- weight_zero_points_all_zero: `True`

## Tensor Oracle

- merge_micro_onnx: `/data/ncnn-logs/ai-team/2026-07-04_22-37-43/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE12-C2F-RESIDUAL-CONCAT-COMPLETION-001/run_logs/stage12_merge_oracle.onnx`
- merge_micro_onnx_sha256: `53aa2110a7e0693a1dbd1fd747054238cff02fc2e465ca11944cd68fcf57bbd7`
- concat_q_mismatches_against_ort: `0`
- concat_q_max_abs_diff_u8: `0`

## Small Fixture Checksums

- seeded_add_float_sum: `341.642841054`
- seeded_concat_s8_sum: `-21728`
- seeded_model2_cv2_i32_sum: `-324013`
- gradient_add_float_sum: `246.883987784`
- gradient_concat_s8_sum: `-22728`
- gradient_model2_cv2_i32_sum: `103397`

## Decision

The Add and Concat contract is float-domain in the accepted Q/DQ graph. Stage 12
therefore uses an explicit measured float fallback for Add/Concat and a post-Concat
Q/DQ handoff before `/model.2/cv2/conv/Conv`. No integer-domain Add shortcut is
accepted by this oracle.
