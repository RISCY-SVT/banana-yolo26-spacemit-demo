# Stage36 A1 Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Scope

Replayed the Stage36 selected `/model.4` ONNX-cut mode through the real runner API:

```text
mode: Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4
merge-repair CLI: branch1_add_lut_cv2_pipelined4
output_quantize: rvv
thread-branch0: 4
thread-branch1: 4
thread-model4-cv2: 4
warmup: 10
runs: 100
repeats: 5
affinity: taskset -c 0-3
```

## Correctness

```text
status: pass
mismatches: 0
max_abs_diff: 0
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
actual_sha256:   70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
frm_modes: RNE RTZ RDN RUP RMM
affinity_ok: 1
```

## Pre-Candidate Replay

First replay before Stage37 code changes:

```text
mean_total_us: 35629.6
stddev_total_us: 144.062
cv_total_pct: 0.404332
conv_us: 20973.9
conv_share_pct: 58.8665
output_quantize_us: 7127.18
output_quantize_share_pct: 20.0035
branch0_conv_us: 7486.79
branch0_compute_us: 5650.47
branch1_conv_us: 6140.24
branch1_compute_us: 4242.87
model4_cv2_conv_us: 7346.84
model4_cv2_compute_us: 3782.44
thread_overhead_us: 4811.48
attribution_pct: 99.9281
```

## Same-Session Baseline

After implementing the Stage37 candidate, the Stage36 mode was replayed again with the same binary/session so candidate speedups use same-session ratios:

```text
mean_total_us: 35774.4
stddev_total_us: 272.052
min_total_us: 35468.2
max_total_us: 36105.9
cv_total_pct: 0.760466
input_adapter_us: 2436.16
conv_us: 21082.8
activation_requant_us: 2986.17
merge_us: 2131.95
output_quantize_us: 7110.83
thread_overhead_us: 4555.05
correction_us: 2531.0
conv_compute_us: 14095.0
attribution_pct: 99.926
```

## Raw Evidence

```text
log_dir: /data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001
pre_candidate_log: run_logs/board_stage36_replay.log
same_session_baseline_log: run_logs/board_stage37_same_session_baseline.log
```

## Non-Claims

This is selected `/model.4` ONNX-cut evidence only. It is not full YOLO26 inference, not model FPS, not full-image/camera performance, not COCO/mAP, and not production/default-backend readiness.
