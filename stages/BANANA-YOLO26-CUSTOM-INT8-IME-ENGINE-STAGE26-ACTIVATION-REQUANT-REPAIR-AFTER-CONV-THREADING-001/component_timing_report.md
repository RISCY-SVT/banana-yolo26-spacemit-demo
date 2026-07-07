# Component Timing Report

Protocol:

```text
board: Banana-Pi BPI-F3 / SpacemiT K1X
cpus: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
```

| path | total_us | stddev_us | conv_us | activation_us | branch0_act_us | branch1_act_us | merge_us | output_quantize_us | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage25 C1 replay with instrumentation | 90086.8 | 322.306 | 26547.1 | 32790.8 | 1253.91 | 31536.9 | 21101.5 | 7019.87 | 0 |
| Stage26 A3 branch1 add LUT | 41573.9 | 537.575 | 26762.0 | 3004.46 | 1260.4 | 1744.06 | 2156.81 | 7013.76 | 0 |

Speedups:

```text
activation_bucket_speedup: 10.914x
total_speedup: 2.166x
merge_bucket_speedup_side_effect: 9.784x
```

Post-repair shares:

```text
conv_share_pct: 64.3721
activation_share_pct: 7.22679
merge_share_pct: 5.18788
output_quantize_share_pct: 16.8706
```
