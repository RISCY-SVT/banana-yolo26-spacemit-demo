# stage23_vs_stage24_timing_reconciliation

## Stage23 Accepted Post-Repair Timing

```text
mean_total_us: 137547
conv_share_pct: 37.9583
activation_share_pct: 23.7138
merge_share_pct: 31.494
output_quantize_share_pct: 4.97977
```

## Stage24 Replay Baseline

```text
mean_total_us: 147624
conv_share_pct: 42.0763
activation_share_pct: 22.0935
merge_share_pct: 29.3546
output_quantize_share_pct: 4.69731
```

The Stage24 baseline replay was slower than the Stage23 accepted number. The replay used the same stable protocol and the same selected `/model.4` ONNX-cut path. Stage24 decisions use the Stage24 replay baseline, not the older Stage23 number.

## Stage24 Candidate

```text
mean_total_us: 125229
conv_share_pct: 49.5835
activation_share_pct: 26.0505
merge_share_pct: 16.7325
output_quantize_share_pct: 5.54777
```

Stage24 B3 reduced total time below both the Stage24 replay and the Stage23 accepted total.
