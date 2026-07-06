# selected_repair_benchmark_report

Protocol:

```text
board: Banana-Pi BPI-F3 / SpacemiT K1X / X60
affinity: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
binary: bench_stage23_model4_runner_cut
mode: ime_threaded
output_quantize: rvv
```

| path | mean_total_us | stddev_total_us | cv_total_pct | mean_merge_us | mean_conv_us | mean_activation_requant_us | mean_output_quantize_us | mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage23 replay baseline | 147624 | 113.47 | 0.0768645 | 43334.4 | 62114.7 | 32615.2 | 6934.36 | 0 |
| Stage24 B3 split1 LUT | 125229 | 147.027 | 0.117407 | 20953.9 | 62092.9 | 32622.7 | 6947.4 | 0 |

```text
merge_speedup: 2.06808x
total_speedup: 1.17883x
merge_total_us_reduced: 22380.5
total_us_reduced: 22395
```

Minimum Stage24 Lane B gate:

```text
runner API ONNX cut mismatches=0: pass
max_abs_diff=0: pass
SHA stable: pass
merge_total_us speedup >=1.5x: pass
selected total_us improves over Stage23 replay: pass
RNE/frm sweep pass: pass
host CTest pass: pass
board CPU0-3 pass: pass
```

Good target:

```text
merge_total_us speedup >=2.0x: pass
total_us <= 120000: not met
```

This is selected `/model.4` ONNX-cut timing only, not model FPS.
