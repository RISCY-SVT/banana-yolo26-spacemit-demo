# ONNX Cut Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001
cut_input: /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output
cut_output: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output

## Fixture Files

```text
input_fixture: .deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_synthetic_seeded/model4_cv1_conv_q_u8_nhwc.bin
input_sha256: e4ec6700e37e974e5bf9814b90c415169b5e514ed9554592238dd836f84fdc5b
expected_output_fixture: .deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_synthetic_seeded/model4_cv2_conv_q_u8_expected_nhwc.bin
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Runner API Status

The replay and candidates used `y26_stage16_model4_c2f_run_cut_u8_output` through `bench_stage23_model4_runner_cut`.

```text
runner_api_status: pass
same_input_onnx_cut_status: pass
mismatches: 0
max_abs_diff: 0
selected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

The Stage25 selected C1 path preserves the Stage22/Stage23 same-input ONNX-cut byte output.
