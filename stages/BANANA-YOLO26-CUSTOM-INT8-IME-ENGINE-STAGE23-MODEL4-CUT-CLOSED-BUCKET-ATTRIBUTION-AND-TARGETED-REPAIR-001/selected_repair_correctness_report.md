# Selected Repair Correctness Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

selected_repair_lane: `D1_output_quantize_rvv`

## Implementation

```text
function: y26_conv_output_quantize_i32_to_u8_rvv_f32
input: /model.4/cv2 corrected int32 output
output: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output uint8 NHWC
output_scale: 0.0660646632314
output_zero_point_u8: 142
rounding: explicit RNE
fallback: y26_conv_output_quantize_i32_to_u8_scalar_unrolled
```

## Host

```text
host_ctest: pass
tests: 37/37
new_test: test_stage23_runner_api_cut
```

## Board

Board `ime_threaded` RVV repair output:

```text
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
actual_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

Board scalar output quantization path also matched the same ONNX cut output:

```text
actual_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Rounding

Ambient `frm` sweep under the real runner API passed for RNE, RTZ, RDN, RUP, and RMM with `mismatches=0` and post-call `frm` restored.

## Conclusion

The selected D1 repair is bit-exact against the same-input ONNX cut output boundary on host and board for the selected `/model.4` cut path.
