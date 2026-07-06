# Stage21 Model4 C2f Integrated Timing Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`
protocol: `warmup=10 runs=100 repeats=5`
board_affinity: `taskset -c 0-3`
board_log: `board_bench_stage21_integrated_stable.log`

## Real Runner Compact Integration Proof

This benchmark uses the real `y26_stage16_model4_c2f_run_*` API and the new merge mode:

```text
Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT
```

| candidate | shape_class | mean_total_us | stddev_total_us | mean_conv_us | mean_activation_requant_us | mean_merge_us | mismatches | affinity_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_real_runner_pre_c2_or_reference | compact_real_runner_oracle_scope | 185.831434 | 0.516033 | 122.315740 | 27.489828 | 22.701748 | 0 | 1 |
| A4_real_runner_threaded_pre_c2 | compact_real_runner_oracle_scope | 289.286412 | 2.817400 | 219.463074 | 29.497970 | 23.335804 | 0 | 1 |
| C2_integrated_model4_c2f | compact_real_runner_oracle_scope | 284.289158 | 0.730216 | 218.501106 | 31.662926 | 17.561526 | 0 | 1 |

Compact threaded timing is not representative/full-shape performance. It proves the integrated mode is callable, bit-correct, and reduces the compact merge bucket from `23.335804 us` to `17.561526 us`.

## Representative/Full-Shape Transfer Timing

The Stage20-compatible full-shape timing matrix was rerun after Stage21 integration:

| candidate | shape_class | mean_total_us | stddev_total_us | mean_conv_us | mean_activation_requant_us | mean_merge_us | mismatches | affinity_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C2_split0_concat_lut_4t | representative_full_shape_model4_c2f_synthetic | 116631 | 364.855 | 53165.3 | 29787.3 | 29767 | 0 | 1 |

Stage20 baseline:

```text
stage20_c2_mean_total_us: 116338
stage20_c2_stddev_total_us: 121.933
stage20_c2_mean_merge_us: 29791.6
```

Stage21 transfer gate:

```text
+3% threshold: 119828 us
+10% threshold: 127972 us
observed: 116631 us
status: pass
```
