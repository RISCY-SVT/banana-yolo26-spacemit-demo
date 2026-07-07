# Candidate Selection Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

## T1 Decision Inputs

```text
conv_total_us: 26753.7
conv_compute_us: 18058.5
conv_correction_us: 2478.61
conv_copy_us: 1251.9
thread_overhead_us: 5007.01
output_quantize_us: 7034.03
```

## Selected Candidate

```text
selected_candidate: T2_fused_correction_writeback
reason: corrected-buffer copy/writeback is local, exact, and removable without changing ONNX-cut math.
```

## Rejected/Deferred Candidates

```text
T3_1x1_tile_prepack: deferred; T1 did not prove pack/prepack dominates. Raw MMT4D compute dominates `/model.4/cv2`.
T4_branch_3x3_tile: deferred; branch Conv compute and thread overhead dominate more than a proven pack/layout sub-bucket.
T5_general_prepack_layout: rejected for Stage28; prepack is prepare-time and not visible as hot selected-cut bucket.
vmadot1_2_3: forbidden in Stage28; only future proof lane may be recommended.
```

## Acceptance Criteria

```text
mismatches: 0 required
max_abs_diff: 0 required
output_sha256 unchanged required
FRM sweep pass required
affinity_ok: 1 required
targeted sub-bucket speedup >= 1.2x OR total speedup >= 1.05x
```

T2 meets the targeted sub-bucket gate by eliminating the measured `conv_copy_us` bucket.
