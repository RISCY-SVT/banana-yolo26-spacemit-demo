# STAGE23_FINAL_REPORT

classification: `stage23-runner-api-onnx-cut-pass-output-quant-repaired-ready-for-next-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `8350c57bd015f044a51800dcd318cb43976e534a`
end_head: `fce411e20eb649e7f7f0cfe65573848c0e8a1fd4`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Summary

Stage23 closed two Stage22 follow-up risks:

```text
onnx_cut_runner_api_status: pass
rounding_regression_status: pass
bucket_attribution_status: pass
selected_repair_lane: D1_output_quantize_rvv
selected_repair_status: pass
host_tests: pass
board_tests: pass
mismatches: 0
max_abs_diff: 0
```

The real model4 C2f runner API now accepts the same full-shape ONNX-cut input tensor and produces the ONNX-cut uint8 output boundary byte-for-byte. The final `/model.4/cv2` output QuantizeLinear bucket was isolated and repaired with an explicit-RNE RVV path.

## Stable Timing

Protocol:

```text
board: svt@banana
affinity: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
```

| path | mean_total_us | stddev_total_us | cv_total_pct | output_quantize_us | attribution_pct | mismatches |
|---|---:|---:|---:|---:|---:|---:|
| Stage22 selected cut | 225214 | 44.6982 | 0.019847 | not-attributed | partial | 0 |
| Stage23 scalar output quant | 205098 | 179.892 | 0.0877103 | 73983.9 | 99.9928 | 0 |
| Stage23 RVV output quant | 137547 | 81.7884 | 0.0594623 | 6849.5 | 99.9892 | 0 |

```text
output_quantize_speedup_scalar_to_rvv: 10.8014x
total_speedup_scalar_to_rvv: 1.4911x
total_speedup_vs_stage22: 1.6373x
```

Post-repair bucket shares:

```text
conv_share_pct: 37.9583
activation_share_pct: 23.7138
merge_share_pct: 31.494
output_quantize_share_pct: 4.97977
```

## Proven

- The real runner API `y26_stage16_model4_c2f_run_cut_u8_output` closes the same-input ONNX cut boundary with `mismatches=0`.
- Host scalar path and board `ime_threaded` path both match `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`.
- Board output SHA256 matches the ONNX cut expected binary:

```text
70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

- Ambient `frm` sweep RNE/RTZ/RDN/RUP/RMM passes through the real runner API, with post-call `frm` restored.
- Bucket attribution is greater than 99.98%.
- Final output quantization is no longer the selected-cut bottleneck.

## Broken

- No remaining Stage23 correctness blocker was observed.
- Stage22 had an unattributed selected-cut bucket and a bench-scoped rounding issue; Stage23 repaired both for the real runner API path.

## Unknown

- Full YOLO26 model correctness remains unknown.
- Full-image/camera performance remains unknown.
- COCO/mAP remains unknown.
- Production/default-backend readiness remains unknown.
- The next selected-cut bottleneck is not yet repaired: Conv is 37.96%, merge is 31.49%, activation is 23.71%.

## Validation Status

```text
host_ctest: pass (37/37)
riscv_cross_build: pass
board_runner_api_vs_onnx_cut: pass
board_rounding_regression: pass
board_stable_benchmark: pass
git_diff_check: pass
symlink_scan: pass
secret_like_scan: pass
result_packet: /exchange/results/outbox/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001
```

## Files Created/Modified

```text
custom_int8_engine/include/y26_k1x_activation.h
custom_int8_engine/kernels/activation_requant.cpp
custom_int8_engine/include/y26_k1x_model4_c2f_runner.h
custom_int8_engine/src/model4_c2f_runner.cpp
custom_int8_engine/tools/bench_stage23_model4_runner_cut.cpp
custom_int8_engine/tests/test_stage23_runner_api_cut.cpp
custom_int8_engine/CMakeLists.txt
custom_int8_engine/tests/CMakeLists.txt
stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001/STAGE22_FINAL_REPORT.md
stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001/STAGE22_SUMMARY_RU.md
stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001/*
```

## Next

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE24-BRANCH1-ACTIVATION-LUT-OR-MERGE-REPAIR-001`

Stage24 should replay Stage23 through the real runner API and choose exactly one next local lane. Current evidence favors merge/post-Concat dataflow review before graph expansion, with branch1 activation and Conv propagation as measured alternatives.

## Non-Claims

This stage is not full YOLO26 inference, not model FPS, not full-image/camera performance, not COCO/mAP, not production readiness, and not default-backend readiness.
