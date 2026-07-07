# Stage26 Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001
replayed_mode: Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT
subset: model4_same_input_onnx_cut

## Protocol

```text
board: Banana-Pi BPI-F3 / SpacemiT K1X
affinity: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
mode: ime_threaded
output_quantize: rvv
merge_repair: branch1_add_lut
thread_branch0: 4
thread_branch1: 4
thread_model4_cv2: 4
```

## Correctness

```text
status: 0
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
```

## FRM Sweep

```text
RNE: pass, post-call frm restored
RTZ: pass, post-call frm restored
RDN: pass, post-call frm restored
RUP: pass, post-call frm restored
RMM: pass, post-call frm restored
```

## Timing

```text
mean_total_us: 41669.2
stddev_total_us: 159.719
cv_total_pct: 0.383302
mean_attribution_pct: 99.9472
mean_other_us: 21.9888
```

Bucket summary:

```text
input_adapter_us: 2615.76
conv_us: 26869.6
activation_requant_us: 2998.92
merge_us: 2088.94
output_quantize_us: 7073.94
thread_overhead_us: 4980.87
correction_us: 3773.79
```

Shares:

```text
conv_share_pct: 64.4832
activation_share_pct: 7.19698
merge_share_pct: 5.01316
output_quantize_share_pct: 16.9765
```

Raw log:

```text
stage26_replay_raw.log
```

This is selected `/model.4` ONNX-cut timing only. It is not full YOLO26 inference, model FPS, full-image/camera performance, COCO/mAP, production readiness, or default backend evidence.
