# Per-Conv Roofline Stage28

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

## Selected Path After T2 Candidate

| node | shape | kernel | MACs | conv_us | GMAC/s | rough_pct_of_2TOPS_equiv | thread_count | bottleneck_class |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `/model.4/m.0/cv1/conv/Conv` | `80x80x32 -> 80x80x16` | `3x3 stride1 pad1` | 29491200 | 7797.29 | 3.78224 | 0.378224% | 4 | `MMT4D_compute_with_thread_overhead` |
| `/model.4/m.0/cv2/conv/Conv` | `80x80x16 -> 80x80x32` | `3x3 stride1 pad1` | 29491200 | 5996.58 | 4.91799 | 0.491799% | 4 | `MMT4D_compute_with_thread_overhead` |
| `/model.4/cv2/conv/Conv` | `80x80x96 -> 80x80x128` | `1x1 stride1 pad0` | 78643200 | 11461.5 | 6.86150 | 0.686150% | 4 | `MMT4D_compute_dominant_structural_low_utilization` |

## Aggregate

```text
aggregate_conv_us: 25255.4
aggregate_macs: 137625600
aggregate_gmac_s: 5.44935
rough_pct_of_2TOPS_equiv: 0.544935%
```

## Interpretation

The T2 candidate removes the explicit copy/writeback sub-bucket but does not address the dominant raw MMT4D compute time. The selected path remains structurally low-utilization for these low-K/threaded MMT4D shapes.

This report is selected-cut diagnostic evidence only, not a full-model roofline.
