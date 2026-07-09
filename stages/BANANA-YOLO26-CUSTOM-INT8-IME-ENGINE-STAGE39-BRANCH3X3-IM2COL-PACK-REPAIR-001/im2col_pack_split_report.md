# Im2col/Pack Split Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


Current Stage38/Stage39 instrumentation measures the fused A-panel preparation as `im2col_pack_us`. The selected runner does not materialize a separate im2col tensor, so exact `im2col_gather_us` versus `packA_us` is not separable without adding intrusive timing around very small inner-loop fragments. For Stage39 the non-overlapping decision bucket is therefore:

```text
im2col_gather_us: 0 (no standalone materialized im2col buffer in selected path)
packA_us: measured fused A-panel gather/pack time
packB_or_weight_pack_us: prepare-time only, not hot-loop bucket
edge_path_us/interior_path_us: implemented as a single fast path with edge fallback; not timed separately to avoid per-panel timing distortion
```

## Measured Fused A-Panel Pack

| bucket | Stage38 replay us | Stage39 fastpack us | speedup |
|---|---:|---:|---:|
| branch0_im2col_pack_us | 3589.480 | 3446.860 | 1.041377x |
| branch1_im2col_pack_us | 1968.590 | 1908.960 | 1.031237x |
| combined_branch3x3_im2col_pack_us | 5558.070 | 5355.820 | 1.037763x |

The Stage39 fastpack candidate did not meet the requested `1.30x` combined im2col/pack speed gate. It did, however, reduce combined branch 3x3 conv total and selected-cut total.
