# Stage27 Final Report

classification: `stage27-conv-decision-selected-tile-prepack-future-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6a32c904cf711afab24e3efd9d2adaa9306c101f`
end_head: `see-final-head-copy-after-local-commit`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
selected_lane: `SELECT_C4_TILE_PREPACK_FUTURE_STAGE`
implemented_candidate: `none`
selected_subset: `model4_same_input_onnx_cut`

## Summary

Stage27 replayed the Stage26 accepted `/model.4` same-input ONNX-cut path through the real runner API:

```text
mode: Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT
taskset: CPU0-3
warmup: 10
runs: 100
repeats: 5
```

Replay correctness passed:

```text
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass for RNE/RTZ/RDN/RUP/RMM
affinity_ok: 1
```

Stable replay timing:

```text
mean_total_us: 41669.2
stddev_total_us: 159.719
cv_total_pct: 0.383302
mean_attribution_pct: 99.9472
```

Post-Stage26 buckets:

```text
conv_us: 26869.6
activation_requant_us: 2998.92
merge_us: 2088.94
output_quantize_us: 7073.94
thread_overhead_us: 4980.87
```

Shares:

```text
conv_share_pct: 64.4832
activation_share_pct: 7.19698
merge_share_pct: 5.01316
output_quantize_share_pct: 16.9765
```

## Conv Decision

Current all-4 cluster0 threading remains the fastest measured local thread-count policy:

```text
true single-thread total_us: 91636.9
current all-4 total_us: 41642.8
```

All tested reduced thread-count variants were slower than all-4 for the selected cut. The 1-thread threaded wrapper was worse than true single-thread IME, so no per-node threshold policy was selected for this current path.

The current threaded Conv path remains low-utilization:

```text
/model.4/m.0/cv1/conv/Conv threaded_GMAC_s: 3.779804
/model.4/m.0/cv2/conv/Conv threaded_GMAC_s: 4.545240
/model.4/cv2/conv/Conv threaded_GMAC_s: 6.251993
```

The selected next lane is a narrow MMT4D/tile/prepack/correction stage, not graph expansion and not `vmadot1/2/3` implementation.

## Broken

No new correctness break was found.

No low-risk Stage27 implementation candidate was accepted because:

```text
current threaded workers are already persistent per node
thread-count reductions regressed timing
output quantize is below the Stage27 secondary threshold
vmadot1/2/3 is forbidden in Stage27
```

## Proven

```text
Stage26 A3 replay correctness: pass
FRM robustness: pass
bucket attribution: pass, 99.9472%
host CTest: pass, 39/39
RISC-V cross build: pass
board stable benchmark: pass
board CPU0-3 affinity: pass
```

## Unknown

```text
exact pack/im2col/correction split inside each Conv node
best MMT4D tile/blocking candidate
whether vmadot1/2/3 can beat current threaded MMT4D on a real 3x3 Conv node
full YOLO26 inference performance
full-image/camera performance
COCO/mAP
production/default backend readiness
```

## Next

Recommended next stage:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001
```

Track B should run separately:

```text
BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
```

This Stage27 report makes no full-model FPS, full-image/camera, COCO/mAP, production, or default-backend claim.
