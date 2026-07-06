# Engine vs ONNX Full-Shape Cut Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## Host Check

```text
binary: .deps/custom_int8_engine/build-host-native-stage22/bench_stage22_model4_onnx_cut
mode: scalar
fixture_dir: .deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_synthetic_seeded
status: pass
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
```

## Board Check

```text
binary: /home/svt/yolo26-custom-int8-stage22/2026-07-06_13-36-27/bench_stage22_model4_onnx_cut
mode: ime_threaded
affinity: taskset -c 0-3
fixture_dir: fixtures
status: pass
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
affinity_ok: 1
```

The board dumped output SHA256 matches the ONNX cut expected NHWC binary exactly:

```text
engine_board_model4_cv2_q_u8_nhwc.bin: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
model4_cv2_conv_q_u8_expected_nhwc.bin: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

Conclusion:

```text
engine_vs_onnx_fullshape_cut: pass
accepted_boundary: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
```
