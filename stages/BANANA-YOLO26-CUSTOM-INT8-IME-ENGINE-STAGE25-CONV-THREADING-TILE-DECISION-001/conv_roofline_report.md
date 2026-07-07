# Conv IME Roofline Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001
scope: selected `/model.4` ONNX-cut path only

## Selected C1 Roofline

| node | MACs | selected_us | effective_GMAC_s | effective_TOPS | percent_of_2TOPS | bottleneck_class |
|---|---:|---:|---:|---:|---:|---|
| `/model.4/m.0/cv1/conv/Conv` | 29491200 | 7830.04 | 3.7664 | 0.003766 | 0.1883 | structural_low_K_or_thread_overhead |
| `/model.4/m.0/cv2/conv/Conv` | 29491200 | 6296.98 | 4.6834 | 0.004683 | 0.2342 | structural_low_K_or_thread_overhead |
| `/model.4/cv2/conv/Conv` | 78643200 | 12037.1 | 6.5334 | 0.006533 | 0.3267 | low_utilization_1x1_mmt4d |

These are selected-cut diagnostics only, not full-model utilization claims.

## Interpretation

The plain `smt.vmadot` MMT4D path remains low-utilization in absolute 2 TOPS terms, but Stage25 C1 threading materially reduced the selected Conv bucket. After C1, activation/requant is the largest bucket, so Stage26 should not start a broad Conv rewrite before repairing activation/requant or confirming that activation cannot be improved locally.
