# Selected Lane Benchmark Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Protocol

```text
board: Banana-Pi BPI-F3 / SpacemiT K1X / X60
affinity: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
timing: accepted runner steady timing, no rdcycle
```

## Timing Table

| metric | Stage36 baseline | Stage37 candidate |
| --- | ---: | ---: |
| mode | branch1_add_lut_cv2_pipelined4 | branch3x3_pipelined4 |
| mean_total_us | 35774.4 | 32307.4 |
| stddev_total_us | 272.052 | 112.917 |
| cv_total_pct | 0.760466 | 0.349509 |
| conv_us | 21082.8 | 17742.7 |
| activation_requant_us | 2986.17 | 2993.66 |
| merge_us | 2131.95 | 2099.23 |
| output_quantize_us | 7110.83 | 7081.0 |
| thread_overhead_us | 4555.05 | 4354.16 |
| branch0_conv_us | 7537.05 | 6099.89 |
| branch0_compute_us | 5847.28 | 4343.22 |
| branch1_conv_us | 6210.42 | 4141.83 |
| branch1_compute_us | 4409.94 | 2814.39 |
| model4_cv2_conv_us | 7335.33 | 7500.94 |
| model4_cv2_compute_us | 3837.75 | 3802.39 |
| attribution_pct | 99.926 | 99.9188 |

## Speedups

```text
selected_cut_total_speedup: 1.107313x
selected_cut_total_delta_us: 3467.0
combined_branch3x3_compute_speedup: 1.433051x
combined_branch3x3_conv_speedup: 1.342301x
```

## Gate Result

```text
combined 3x3 GEMM/compute speedup >= 1.25x: pass
combined 3x3 GEMM/compute speedup >= 1.40x: pass
selected-cut total speedup >= 1.05x: pass
selected-cut total speedup >= 1.10x: pass
```

## Post-Candidate Bottleneck

After Stage37:

```text
conv_share_pct: 54.9182
output_quantize_share_pct: 21.9176
activation_requant_share_pct: 9.26617
merge_share_pct: 6.49766
```

The next stage should replay the Stage37 selected mode and choose between remaining Conv work and output QuantizeLinear repair from fresh same-session evidence.
