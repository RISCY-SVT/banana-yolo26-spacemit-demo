# Model4 C2f ONNX Cut Boundary Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## Boundary

The Stage22 same-input closure uses the quantized `/model.4` C2f boundary that matches the Stage20/Stage21 full-shape representative runner input.

Input tensor:

```text
onnx_name: /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output
shape_nchw: 1x64x80x80
shape_nhwc_for_cpp: 1x80x80x64
dtype: uint8 QDQ code
cpp_storage: converted to signed int8 storage by code - 128
scale: 0.0688334777951
zero_point_u8: 163
input_nhwc_bin: .deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_synthetic_seeded/model4_cv1_conv_q_u8_nhwc.bin
input_nhwc_bin_sha256: e4ec6700e37e974e5bf9814b90c415169b5e514ed9554592238dd836f84fdc5b
```

Output tensor:

```text
onnx_name: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
shape_nchw: 1x128x80x80
shape_nhwc_for_cpp: 1x80x80x128
dtype: uint8 QDQ code
scale: 0.0660646632314
zero_point_u8: 142
expected_nhwc_bin: .deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_synthetic_seeded/model4_cv2_conv_q_u8_expected_nhwc.bin
expected_nhwc_bin_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

Accepted ONNX model:

```text
path: .deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
sha256: 30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
opset: ai.onnx 18
```

## C++ Runner Entry

Stage22 added `bench_stage22_model4_onnx_cut`, a same-input verifier/bench that runs the integrated `/model.4` C2f downstream path from the quantized `/model.4/cv1/conv` boundary and compares the final `/model.4/cv2/conv` quantized output against the ONNX cut oracle.

The verifier uses the same Stage20/Stage21 selected C2 merge semantics:

```text
selected_merge_mode: C2_SPLIT0_CONCAT_LUT
selected_conv_mode: ime_threaded for board timing
cluster_policy: CPU0-3 only
forbidden: CPU4-7 IME
```

This is not a full engine API, graph scheduler, full-image path, model FPS path, camera path, COCO/mAP path, or production/default backend.
