# Conv Component Split Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

## Instrumentation

Stage28 added non-overlapping critical-worker component timing to the threaded Conv path:

```text
worker_compute_us: raw MMT4D IME call for the critical worker
worker_correction_us: u8-as-s8 zero-point correction and bias add for rows written by the critical worker
worker_copy_us: old corrected-buffer copy/writeback path
worker_other_us: critical worker total minus compute/correction/copy
thread_overhead_us: whole threaded region total minus critical worker total
```

Threaded workers are still cluster0-only and use the same selected runner API. No math, Q/DQ, activation, merge, or output-quantize semantics were changed by T1 instrumentation.

## Stage27 Replay Split

| bucket | mean_us |
| --- | ---: |
| total | 41580.9 |
| conv_total | 26753.7 |
| conv_compute_critical_path | 18058.5 |
| conv_correction_critical_path | 2478.61 |
| conv_copy_critical_path | 1251.9 |
| conv_worker_other | 3.88932 |
| conv_thread_overhead | 5007.01 |
| activation_requant | 2999.07 |
| merge | 2148.66 |
| output_quantize | 7034.03 |

## Per-Conv Stage27 Replay Split

| node | shape | kernel | mac_count | conv_us | compute_us | correction_us | copy_us | inferred_thread_overhead_us | bottleneck |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/model.4/m.0/cv1/conv/Conv` | `80x80x32 -> 80x80x16` | `3x3` | 29491200 | 7718.13 | 5652.96 | 233.884 | 61.1128 | 1769.52 | `MMT4D_compute_with_thread_overhead` |
| `/model.4/m.0/cv2/conv/Conv` | `80x80x16 -> 80x80x32` | `3x3` | 29491200 | 6324.68 | 4312.3 | 521.524 | 158.699 | 1330.82 | `MMT4D_compute_with_thread_overhead` |
| `/model.4/cv2/conv/Conv` | `80x80x96 -> 80x80x128` | `1x1` | 78643200 | 12710.9 | 8093.26 | 1723.2 | 1032.09 | 1859.65 | `MMT4D_compute_with_copy_material` |

## Decision From T1

T1 shows raw MMT4D compute is the largest Conv sub-bucket, but the corrected-buffer writeback/copy bucket is locally removable and material enough for a bounded T2 candidate. Stage28 therefore selected exactly one local repair lane:

```text
selected_candidate: T2_fused_correction_writeback
```
