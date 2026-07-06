# Runner API vs ONNX Cut Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

## Boundary

```text
runner_api: y26_stage16_model4_c2f_run_cut_u8_output
input_tensor: /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output
input_dtype: uint8
input_shape_nchw: 1x64x80x80
input_shape_nhwc: 1x80x80x64
input_scale: 0.0688334777951
input_zero_point_u8: 163
input_nhwc_bin_sha256: e4ec6700e37e974e5bf9814b90c415169b5e514ed9554592238dd836f84fdc5b

output_tensor: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
output_dtype: uint8
output_shape_nchw: 1x128x80x80
output_shape_nhwc: 1x80x80x128
output_scale: 0.0660646632314
output_zero_point_u8: 142
onnx_cut_expected_nhwc_bin_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Model

```text
accepted_qdq_model: .deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
accepted_qdq_model_sha256: 30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
onnx_cut_model_sha256: bde82b0130615717ffcbdbaca8fa274e5de00c111cf0b0a518023b6a674d841a
fixture_dir: .deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_synthetic_seeded
```

## Closure Result

The Stage23 bench is a thin wrapper around the real local model4 C2f runner API:

```text
y26_stage16_model4_c2f_prepare_cut_threaded_branch0
y26_stage16_model4_c2f_run_cut_u8_output
```

It no longer duplicates the full pipeline as local bench-only composition. The runner API exposes the cut input and uint8 ONNX-cut output boundary directly.

Host scalar correctness:

```text
mode: scalar
output_quantize: scalar
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
```

Board threaded IME correctness:

```text
mode: ime_threaded
output_quantize: rvv
affinity: taskset -c 0-3
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
actual_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Conclusion

`H1_runner_api_cut_closure` passed. The real integrated runner API can consume the same full-shape ONNX cut input tensor and produces byte-identical output for the selected `/model.4` C2f cut boundary.

This is selected-cut evidence only. It is not full YOLO26 inference, not full-image/camera performance, not COCO/mAP, not model FPS, and not production/default-backend evidence.
