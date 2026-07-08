# Stage34 Baseline Replay Report

selected_path: `Stage26 A3 branch1/add LUT + Stage25 C1 threaded Conv policy`
mode: `ime_threaded`
merge_repair: `branch1_add_lut`
output_quantize: `rvv`
thread_branch0: `4`
thread_branch1: `4`
thread_model4_cv2: `4`
protocol: `taskset -c 0-3`, `warmup=10 runs=100 repeats=5`

Raw log:

```text
/data/ncnn-logs/ai-team/2026-07-08_16-50-48/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-VMADOT-CV2-PIPELINED-MICROKERNEL-001/run_logs/board_stage34_baseline_replay.out
```

## Correctness

```text
status: 0
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
affinity_ok: 1
frm_sweep: pass for RNE/RTZ/RDN/RUP/RMM
```

## Timing

```text
mean_total_us: 40178.5
stddev_total_us: 283.996
cv_total_pct: 0.706837
mean_input_adapter_us: 2392.63
mean_conv_us: 25585.8
mean_activation_requant_us: 2980.93
mean_merge_us: 2124
mean_output_quantize_us: 7070.4
mean_thread_overhead_us: 5243.32
mean_correction_us: 2462.31
mean_conv_compute_us: 17923
mean_attribution_pct: 99.9385
```

## `/model.4/cv2/conv/Conv`

```text
mean_model4_cv2_conv_us: 12096.5
mean_model4_cv2_correction_us: 1753.73
mean_model4_cv2_compute_us: 8071.68
mean_model4_cv2_copy_us: 0
mean_model4_cv2_worker_other_us: 0.631618
```

This is selected `/model.4` ONNX-cut timing only, not full YOLO26 inference, not model FPS, and not full-image/camera performance.
