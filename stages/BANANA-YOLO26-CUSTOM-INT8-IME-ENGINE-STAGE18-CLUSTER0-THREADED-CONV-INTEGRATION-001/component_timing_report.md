# Component Timing Report

Protocol:

```text
warmup: 10
runs: 100
repeats: 5
pinning: taskset -c 0-3
scope: representative/full-shape /model.4 branch-entry selected subset
```

| candidate | threads | mean_total_us | stddev_total_us | mean_conv_us | stddev_conv_us | mean_activation_requant_us | mean_split_us | conv_share_pct | activation_share_pct | total_speedup_vs_A0 | conv_speedup_vs_A0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_stage17_single_thread_replay | 1 | 25588.395318 | 3.740899 | 20414.513636 | 10.350443 | 4985.479680 | 187.070140 | 79.780359 | 19.483362 | 1.000000 | 1.000000 |
| A1_integrated_threaded_conv_1t | 1 | 25982.870290 | 11.103531 | 20792.320188 | 8.233628 | 4990.830640 | 195.773436 | 80.023184 | 19.208157 | 0.984818 | 0.981830 |
| A2_integrated_threaded_conv_2t | 2 | 16348.834112 | 155.650231 | 11194.148808 | 151.828973 | 4948.961570 | 201.646556 | 68.470624 | 30.271037 | 1.565151 | 1.823677 |
| A3_integrated_threaded_conv_3t | 3 | 12711.599524 | 70.819213 | 7503.098110 | 72.434893 | 5011.754118 | 192.177880 | 59.025602 | 39.426621 | 2.012996 | 2.720811 |
| A4_integrated_threaded_conv_4t | 4 | 11211.333822 | 184.481542 | 6025.979842 | 187.863483 | 4990.630782 | 190.598392 | 53.749000 | 44.514157 | 2.282369 | 3.387750 |

Selected mode:

```text
A4_integrated_threaded_conv_4t
```

This is selected-subset microbenchmark evidence only. It is not full YOLO26 FPS, not full-image/camera performance, and not production evidence.
