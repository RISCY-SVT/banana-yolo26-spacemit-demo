# Candidate Benchmark Report

protocol: `taskset -c 0-3`, `warmup=10 runs=100 repeats=5`
scope: selected `/model.4` ONNX-cut path only, not model FPS

| mode | total_us | stddev_us | model4_cv2_conv_us | model4_cv2_correction_us | model4_cv2_compute_us | model4_cv2_copy_us | status |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline `branch1_add_lut` | 40380.4 | 592.759 | 11852.7 | 1742.83 | 8129.4 | 0 | pass |
| candidate `branch1_add_lut_mixed_cv2` | 40934.1 | 446.612 | 12862.2 | 0 | 9699.05 | 1127.18 | correct-regressed |

Derived ratios:

```text
model4_cv2_correction_speedup: eliminated baseline measured correction bucket
model4_cv2_conv_speedup: 0.9215x
selected_cut_total_speedup: 0.9865x
selected_cut_total_regression: 1.37%
```

Acceptance evaluation:

```text
A correction drop >=50% with no total regression >1%: fail
B model4_cv2 total_conv_us improves >=1.05x: fail
C selected-cut total_us improves >=1.02x: fail
```

Candidate decision: correct but regresses; do not select/promote as accepted runtime mode.
