# Per-Conv Attribution Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Conv Nodes

```text
/model.4/m.0/cv1/conv/Conv: 3x3, 80x80x32 -> 80x80x16
/model.4/m.0/cv2/conv/Conv: 3x3, 80x80x16 -> 80x80x32
/model.4/cv2/conv/Conv:      1x1, 80x80x96 -> 80x80x128
```

## Same-Session Baseline

Mode: `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4`

| node | conv_us | compute_us | im2col_pack_us | correction_us | writeback_us | thread_overhead_us | worker_other_us |
| --- | ---: | ---: | --- | ---: | --- | --- | ---: |
| /model.4/m.0/cv1/conv/Conv | 7537.05 | 5847.28 | included_in_compute | 219.926 | included_in_conv | included_in_thread_overhead | 0.173982 |
| /model.4/m.0/cv2/conv/Conv | 6210.42 | 4409.94 | included_in_compute | 522.484 | included_in_conv | included_in_thread_overhead | 0.223938 |
| /model.4/cv2/conv/Conv | 7335.33 | 3837.75 | included_in_compute | 1788.59 | included_in_conv | included_in_thread_overhead | 0.23283 |

## Stage37 Candidate

Mode: `Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4`

| node | conv_us | compute_us | im2col_pack_us | correction_us | writeback_us | thread_overhead_us | worker_other_us |
| --- | ---: | ---: | --- | ---: | --- | --- | ---: |
| /model.4/m.0/cv1/conv/Conv | 6099.89 | 4343.22 | included_in_compute | 253.691 | included_in_conv | included_in_thread_overhead | 0.198256 |
| /model.4/m.0/cv2/conv/Conv | 4141.83 | 2814.39 | included_in_compute | 444.516 | included_in_conv | included_in_thread_overhead | 0.22809 |
| /model.4/cv2/conv/Conv | 7500.94 | 3802.39 | included_in_compute | 1796.0 | included_in_conv | included_in_thread_overhead | 0.294858 |

## Branch 3x3 Gate

```text
combined_branch3x3_compute_baseline_us: 10257.22
combined_branch3x3_compute_candidate_us: 7157.61
combined_branch3x3_compute_speedup: 1.433051x

combined_branch3x3_conv_baseline_us: 13747.47
combined_branch3x3_conv_candidate_us: 10241.72
combined_branch3x3_conv_speedup: 1.342301x

selected_cut_total_baseline_us: 35774.4
selected_cut_total_candidate_us: 32307.4
selected_cut_total_speedup: 1.107313x
```

## Decision Impact

The selected Lane A candidate passes:

```text
combined 3x3 GEMM/compute speedup >= 1.25x: pass
combined 3x3 GEMM/compute speedup >= 1.40x: pass
selected-cut total speedup >= 1.05x: pass
same-input ONNX-cut output unchanged: pass
FRM sweep: pass
```

## Attribution Limitation

The current timing API does not expose a non-overlapping `im2col_pack_us` per Conv node. Stage37 therefore reports `compute_us` as the directly available combined worker compute bucket. No im2col-specific optimization claim is made.
