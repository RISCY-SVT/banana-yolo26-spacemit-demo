# Per-Conv Roofline Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001
subset: model4_same_input_onnx_cut

## Source

The table uses the Stage27 same-session true single-thread IME anchor and the Stage26 accepted threaded replay. The rough 2 TOPS-equivalent percentage treats 2 TOPS as about `1000 GMAC/s`.

See also:

```text
conv_shape_table.tsv
```

## Conv Nodes

| node | shape | kernel | MACs | single_thread_us | threaded_us | speedup | threaded_GMAC/s | approx_2TOPS_pct | classification |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| /model.4/m.0/cv1/conv/Conv | 80x80x32 -> 80x80x16 | 3x3 | 29491200 | 22088.4 | 7802.31 | 2.831x | 3.779804 | 0.377980 | structural_low_K_or_packing_bound |
| /model.4/m.0/cv2/conv/Conv | 80x80x16 -> 80x80x32 | 3x3 | 29491200 | 17750.1 | 6488.37 | 2.736x | 4.545240 | 0.454524 | structural_low_K_or_packing_bound |
| /model.4/cv2/conv/Conv | 80x80x96 -> 80x80x128 | 1x1 | 78643200 | 37331.9 | 12578.9 | 2.968x | 6.251993 | 0.625199 | structural_low_K_or_packing_bound |

## Interpretation

All three Conv nodes benefit strongly from existing cluster0 threading, but effective throughput remains far below the rough IME peak-equivalent. The post-Concat `/model.4/cv2/conv/Conv` is the largest single Conv bucket, and the two 3x3 branch Conv nodes remain collectively material.

This evidence points to a narrow future Conv kernel/tile/prepack/correction stage before any graph expansion. A `vmadot1/2/3` proof lane may be considered later, but Stage27 does not authorize or implement it.
