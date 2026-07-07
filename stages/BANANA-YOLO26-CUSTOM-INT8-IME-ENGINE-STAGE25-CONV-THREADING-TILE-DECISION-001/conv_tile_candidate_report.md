# Conv Tile Candidate Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

## Status

tile_kernel_candidate_status: not_attempted_by_scope

## Reason

Stage25 C1 threading propagation produced a same-session selected-cut total improvement:

```text
Stage24 replay total_us: 125176
Stage25 selected C1 total_us: 89178.9
total_speedup: 1.4037x
```

After C1, Conv is no longer the dominant bucket:

```text
conv_share_pct: 29.3389
activation_share_pct: 36.7808
merge_share_pct: 23.5081
```

Therefore a MMT4D/tile/blocking/prepack rewrite is not the immediate next local repair. It remains a future lane if activation/requant and merge repairs stop being material and Conv becomes dominant again.
