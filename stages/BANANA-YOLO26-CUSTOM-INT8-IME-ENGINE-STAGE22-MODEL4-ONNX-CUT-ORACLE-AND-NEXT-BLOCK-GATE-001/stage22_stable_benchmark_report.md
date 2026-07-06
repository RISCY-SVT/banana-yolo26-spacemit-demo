# Stage22 Stable Benchmark Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

This benchmark is selected `/model.4` C2f same-input ONNX-cut evidence only. It is not full YOLO26 FPS, not full-image/camera performance, not COCO/mAP, and not a production/default backend claim.

## Protocol

```text
board: svt@banana
affinity: taskset -c 0-3
mode: ime_threaded
warmup: 10
runs: 100
repeats: 5
fixture_dir: /home/svt/yolo26-custom-int8-stage22/2026-07-06_13-36-27/fixtures
```

## Result

```text
status: pass
mismatches: 0
max_abs_diff: 0
checksum: 106597930
expected_checksum: 106597930
affinity_ok: 1
mean_total_us: 225214
stddev_total_us: 44.6982
min_total_us: 225147
max_total_us: 225259
cv_total_pct: 0.019847
mean_conv_us: 52062.6
mean_activation_requant_us: 32388.1
mean_merge_us: 42446.9
mean_thread_overhead_us: 352.271
mean_correction_us: 2371.64
mean_branch0_conv_us: 5860.34
mean_branch1_conv_us: 15747.2
mean_model4_cv2_conv_us: 30455
conv_share_pct: 23.1169
activation_share_pct: 14.381
merge_share_pct: 18.8474
```

The stable dumped board output SHA256 matches the ONNX cut expected binary:

```text
engine_board_model4_cv2_q_u8_nhwc_stable.bin: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
model4_cv2_conv_q_u8_expected_nhwc.bin: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```
