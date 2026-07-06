# runner_api_vs_onnx_cut_report

## Boundary

input: `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output`
output: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`
runner_api: `y26_stage16_model4_c2f_run_cut_u8_output`
selected_mode: `Y26_STAGE16_MERGE_MODE_STAGE24_B3_SPLIT1_LUT`
output_quantize: `rvv`

## Board Same-Input Result

```text
status: 0
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
actual_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

Host compact CTest also proved that the B3 runner path matches the baseline C2 path:

```text
test_stage24_merge_repair: pass
concat_mismatches: 0
output_mismatches: 0
```

Status: `pass`.
