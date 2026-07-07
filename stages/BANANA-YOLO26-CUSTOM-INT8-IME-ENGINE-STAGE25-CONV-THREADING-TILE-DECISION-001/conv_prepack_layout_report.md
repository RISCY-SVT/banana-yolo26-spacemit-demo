# Conv Prepack And Layout Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

## Current State

The selected `/model.4` cut path reuses persistent MMT4D prepack and per-run reusable workspaces. Stage25 did not change weight packing math or introduce a new pack/layout format.

## Layout Buckets

The Stage25 selected C1 benchmark reports:

```text
mean_copy_layout_us: 0
mean_pack_layout_us: 0
mean_attribution_pct: 99.977
```

Conv timing includes the existing MMT4D pack/im2col/workspace behavior inside the Conv kernels. No non-Conv layout bucket dominated the selected path after Stage24 merge repair.

## Decision

No prepack/layout candidate was selected in Stage25. If future per-Conv instrumentation separates packA/im2col from compute and shows pack dominates, open a narrow pack/layout stage for the specific Conv node.
