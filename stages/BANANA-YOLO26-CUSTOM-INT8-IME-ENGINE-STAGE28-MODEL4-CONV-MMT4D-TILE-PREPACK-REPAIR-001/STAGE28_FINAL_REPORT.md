# Stage28 Final Report

classification: `stage28-conv-component-split-complete-candidate-selected-and-accepted`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `502f7abe06aaba413310731971176ede603f527f`
end_head: `see-final-head-copy-after-local-commit`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
selected_subset: `model4_same_input_onnx_cut`
selected_candidate: `T2_fused_correction_writeback`

## Summary

Stage28 replayed the Stage26/27 selected `/model.4` same-input ONNX-cut runner path and added mandatory Conv component decomposition before selecting a candidate.

Replay gate passed:

```text
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass for RNE RTZ RDN RUP RMM
affinity_ok: 1
```

T1 component split showed:

```text
total_us: 41580.9
conv_us: 26753.7
conv_compute_us: 18058.5
conv_correction_us: 2478.61
conv_copy_us: 1251.9
thread_overhead_us: 5007.01
```

Because the corrected-buffer copy/writeback was local and exact, Stage28 selected one candidate:

```text
T2_fused_correction_writeback
```

T2 writes corrected rows directly into the final output buffer and skips correction/copy for overcomputed halo rows. The correction formula is unchanged.

## T2 Result

```text
mean_total_us_before: 41580.9
mean_total_us_after: 40231.6
total_speedup: 1.03354x
mean_conv_us_before: 26753.7
mean_conv_us_after: 25255.4
conv_speedup: 1.05933x
conv_copy_us_before: 1251.9
conv_copy_us_after: 0
```

Correctness remained exact:

```text
mismatches: 0
max_abs_diff: 0
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
expected_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
affinity_ok: 1
```

## Post-T2 Buckets

```text
total_us: 40231.6
conv_us: 25255.4
activation_requant_us: 2983.1
merge_us: 2180.94
output_quantize_us: 7071.88
thread_overhead_us: 4705.99
conv_compute_us: 18097.1
conv_correction_us: 2500.86
conv_copy_us: 0
```

Shares:

```text
conv_share_pct: 62.775
activation_share_pct: 7.41481
merge_share_pct: 5.42096
output_quantize_share_pct: 17.5779
```

## Conv Roofline

| node | conv_us | GMAC/s | bottleneck |
| --- | ---: | ---: | --- |
| `/model.4/m.0/cv1/conv/Conv` | 7797.29 | 3.78224 | `MMT4D_compute_with_thread_overhead` |
| `/model.4/m.0/cv2/conv/Conv` | 5996.58 | 4.91799 | `MMT4D_compute_with_thread_overhead` |
| `/model.4/cv2/conv/Conv` | 11461.5 | 6.86150 | `MMT4D_compute_dominant_structural_low_utilization` |

## Broken

No correctness regression was found.

T2 does not solve the dominant raw MMT4D compute bucket. After T2, Conv remains the largest selected-cut bucket.

## Proven

```text
same-input ONNX-cut runner correctness: pass
FRM robustness: pass
component split attribution: pass
selected T2 candidate correctness: pass
host CTest: pass, 39/39
RISC-V cross build: pass
board stable benchmark: pass
```

## Unknown

```text
best direct-conv or vmadot1/2/3 applicability
full YOLO26 inference performance
full-image/camera performance
COCO/mAP
production/default backend readiness
YOLO26 model-value vs YOLO11 production baseline
```

## Next

Recommended next stage:

```text
BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
```

Reason: Stage28 now shows structural low utilization in selected-cut Conv after the local T2 repair, but the prompt requires any major `vmadot1/2/3` investment to be gated by Track B YOLO26 mAP/value evidence.

## Non-Claims

This is not full YOLO26 inference. This is not model FPS. This is not full-image/camera performance. This is not COCO/mAP. This is not production/default-backend readiness.
