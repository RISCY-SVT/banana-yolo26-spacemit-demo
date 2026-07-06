# Selected Repair Benchmark Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

This is selected `/model.4` C2f same-input ONNX-cut evidence only. It is not full YOLO26 FPS, not full-image/camera performance, not COCO/mAP, and not production/default-backend evidence.

## Protocol

```text
board: svt@banana
affinity: taskset -c 0-3
mode: ime_threaded
warmup: 10
runs: 100
repeats: 5
fixture_dir: /home/svt/yolo26-custom-int8-stage23/2026-07-06_15-39-22/fixtures
```

## Results

| path | mean_total_us | stddev_total_us | cv_total_pct | output_quantize_us | mismatches | checksum |
|---|---:|---:|---:|---:|---:|---:|
| scalar output quantize | 205098 | 179.892 | 0.0877103 | 73983.9 | 0 | 106597930 |
| RVV output quantize | 137547 | 81.7884 | 0.0594623 | 6849.5 | 0 | 106597930 |

## Speedups

```text
output_quantize_speedup_scalar_to_rvv: 10.8014x
total_speedup_scalar_to_rvv: 1.4911x
total_speedup_vs_stage22_stable_total_225214: 1.6373x
```

## RVV Bucket Shares After Repair

```text
conv_share_pct: 37.9583
activation_share_pct: 23.7138
merge_share_pct: 31.494
output_quantize_share_pct: 4.97977
```

## Acceptance

```text
output_quantize bucket speedup >=3x: pass
full selected /model.4 cut total improves vs scalar output quantize baseline: pass
full selected /model.4 cut total improves vs Stage22 total: pass
runner API vs ONNX cut still mismatches=0: pass
```
