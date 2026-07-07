# Thread Overhead Stage28

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

## Definition

```text
thread_overhead_us = threaded_region_total_us - critical_worker_total_us
```

This includes main-thread barrier wait, worker skew, handoff overhead, and any non-worker time in the selected threaded Conv regions. It does not include CPU4-7; worker affinity reported `affinity_ok=1`.

## Same-Session Results

| path | total_us | conv_us | thread_overhead_us | thread_overhead_share_total | thread_overhead_share_conv |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage27 replay T1 | 41580.9 | 26753.7 | 5007.01 | 12.0416% | 18.7152% |
| Stage28 T2 | 40231.6 | 25255.4 | 4705.99 | 11.6973% | 18.6336% |

## Interpretation

Thread overhead remains material, but Stage27 already showed per-node worker pools are persistent and all-4 threading is the best tested policy. Stage28 did not attempt a broader threading redesign because the prompt scope is Conv/MMT4D component repair and no graph-wide scheduler or default dispatch change is authorized.
