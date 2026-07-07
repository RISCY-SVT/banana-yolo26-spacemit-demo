# Candidate Decision

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

## Selected Lane

```text
SELECT_C4_TILE_PREPACK_FUTURE_STAGE
```

Classification:

```text
stage27-conv-decision-selected-tile-prepack-future-stage
```

## Evidence

Stage26 accepted path replay:

```text
mean_total_us: 41669.2
conv_us: 26869.6
conv_share_pct: 64.4832
output_quantize_share_pct: 16.9765
mismatches: 0
max_abs_diff: 0
frm_sweep: pass
mean_attribution_pct: 99.9472
```

Threading matrix:

```text
current all-4 total_us: 41642.8
true single-thread total_us: 91636.9
all 1-thread threaded wrapper total_us: 94921.7
```

Current 4-thread policy remains the fastest measured local thread-count policy. Per-node thresholding was not selected because all tested 2/3-thread reductions regressed total time.

## Rejected Lanes

`SELECT_C2_PERSISTENT_POOL` rejected for Stage27:

```text
Current implementation already has persistent per-node workers at prepare-time.
Remaining thread_overhead_us is barrier/copy/scheduling overhead inside persistent workspaces.
Cross-node shared workers would be a broader scheduling change.
```

`SELECT_C3_THREAD_THRESHOLD` rejected:

```text
All present Conv nodes are large enough to benefit from 4 threads.
The 1-thread threaded wrapper is slower than true single-thread IME.
```

`SELECT_C5_VMADOT123_FUTURE_PROOF_STAGE` deferred:

```text
Evidence supports future proof-lane consideration, but the current largest Conv is 1x1 and a lower-risk MMT4D/tile/prepack/correction stage should come first.
```

`SELECT_C6_OUTPUT_QUANTIZE_SECONDARY_REPAIR` rejected:

```text
Output quantize share is about 17%, below the Stage27 >20% threshold.
```

`SELECT_NO_LOCAL_REPAIR_TRACK_B_FIRST` rejected:

```text
Custom selected-cut Conv remains a clear local bottleneck. Track B should run in parallel, not replace the next custom-engine stage.
```

## Next

Recommended Stage28:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001
```
