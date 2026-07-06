# Stage22 vs Stage23 Timing Reconciliation

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

## Stage22

```text
mean_total_us: 225214
stddev_total_us: 44.6982
conv_us: 52062.6
activation_requant_us: 32388.1
merge_us: 42446.9
thread_overhead_us: 352.271
correction_us: 2371.64
output_quantize_us: not separately reported
```

Stage22 closed same-input ONNX cut correctness but left a large unattributed timing bucket.

## Stage23 Attribution Baseline

Stage23 replayed the selected runner API with scalar final output quantization:

```text
mean_total_us: 205098
stddev_total_us: 179.892
output_quantize_us: 73983.9
mean_attribution_pct: 99.9928
mismatches: 0
```

The scalar replay is lower than Stage22 total because the Stage23 tool uses a thinner real-runner cut wrapper and cleaner bucket accounting. The final output quantization bucket is now explicit.

## Stage23 Repair

Stage23 selected D1 RVV final output quantization:

```text
mean_total_us: 137547
stddev_total_us: 81.7884
output_quantize_us: 6849.5
mean_attribution_pct: 99.9892
mismatches: 0
```

## Reconciliation

```text
stage22_total_us: 225214
stage23_scalar_output_quant_total_us: 205098
stage23_rvv_output_quant_total_us: 137547
stage23_rvv_total_improvement_vs_stage22: 87667 us
stage23_rvv_total_speedup_vs_stage22: 1.6373x
```

The Stage22 timing gap is reconciled as final output QuantizeLinear plus bench-wrapper overhead. Stage23 repaired the output QuantizeLinear component while preserving same-input ONNX cut equality.
