# Stage26 Final Report

classification: `stage26-activation-requant-repaired-ready-for-next-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `b382bd71c4091cc3476d59f77cb35c2a0d246513`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
selected_subset: `model4_same_input_onnx_cut`
selected_lane: `A3`
accepted_candidate: `A3_branch1_add_lut`

## Summary

Stage26 replayed the Stage25 selected `/model.4` ONNX-cut path and confirmed same-input byte correctness before repair. The replay showed branch1 activation was the dominant activation/requant subbucket:

```text
activation_requant_us: 32790.8
branch0_activation_us: 1253.91
branch1_activation_us: 31536.9
```

Stage26 then integrated an explicit real-runner local mode:

```text
Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT
```

The mode computes branch1 conv uint8 codes with the existing RNE quantize path and uses a prepare-time `split1_code x branch1_conv_code -> concat_s8` LUT for the float-domain Add/post-QDQ slot. This removes per-element branch1 `std::exp` and float add quantization from the hot loop while preserving ONNX-cut bytes.

## Correctness

```text
mismatches: 0
max_abs_diff: 0
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
frm_sweep: pass for RNE/RTZ/RDN/RUP/RMM
```

## Timing

| path | total_us | stddev_us | activation_us | conv_us | merge_us | output_quantize_us | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage25 C1 replay with instrumentation | 90086.8 | 322.306 | 32790.8 | 26547.1 | 21101.5 | 7019.87 | 0 |
| Stage26 A3 branch1 add LUT | 41573.9 | 537.575 | 3004.46 | 26762.0 | 2156.81 | 7013.76 | 0 |

Speedups:

```text
activation_requant_speedup: 10.914x
total_speedup: 2.166x
merge_side_effect_speedup: 9.784x
```

Post-repair shares:

```text
conv_share_pct: 64.3721
activation_share_pct: 7.22679
merge_share_pct: 5.18788
output_quantize_share_pct: 16.8706
```

## Validation

```text
host_pre_ctest: 38/38 passed
host_post_ctest: 39/39 passed
riscv_cross_build_pre: pass
riscv_cross_build_post: pass
board_correctness: pass
board_stable_benchmark: pass
```

## Non-claims

This stage is not full YOLO26 inference, model FPS, full-image/camera performance, COCO/mAP, production readiness, or default backend readiness.

## Next

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001`
