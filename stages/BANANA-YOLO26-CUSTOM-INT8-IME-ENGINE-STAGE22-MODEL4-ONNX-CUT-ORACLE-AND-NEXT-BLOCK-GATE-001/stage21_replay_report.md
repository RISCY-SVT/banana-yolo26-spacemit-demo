# Stage21 Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## Command

```text
board_dir: /home/svt/yolo26-custom-int8-stage22/2026-07-06_13-36-27
command: taskset -c 0-3 ./bench_stage21_model4_c2f_integrated 10 100 5
```

## Result

The available Stage21 replay binary reports compact real-runner oracle-scope timing, not the Stage22 full-shape ONNX-cut timing.

```text
A0_real_runner_pre_c2_or_reference:
  shape_class: compact_real_runner_oracle_scope
  status: pass
  mismatches: 0
  mean_total_us: 195.839154
  stddev_total_us: 1.314604

A4_real_runner_threaded_pre_c2:
  shape_class: compact_real_runner_oracle_scope
  status: pass
  mismatches: 0
  mean_total_us: 298.874504
  stddev_total_us: 5.560310

C2_integrated_model4_c2f:
  shape_class: compact_real_runner_oracle_scope
  status: pass
  mismatches: 0
  mean_total_us: 289.441292
  stddev_total_us: 0.394547
```

Interpretation:

```text
stage21_compact_replay: pass
full_shape_transfer_sanity: not-applicable-to-this-binary
```

Stage22 full-shape same-input timing is reported separately in `stage22_stable_benchmark_report.md`.
