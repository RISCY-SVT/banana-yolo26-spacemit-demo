# Memory Traffic Estimate

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


The selected path uses fused A-panel gather/pack into the MMT4D `4 x K` workspace. It does not write a full materialized im2col tensor. Approximate logical A-panel payload per full 80x80 node, excluding halo overcompute and repeated output-channel reuse, is:

| node | C_in | logical A-panel bytes per full pass |
|---|---:|---:|
| /model.4/m.0/cv1/conv/Conv | 32 | 1843200 |
| /model.4/m.0/cv2/conv/Conv | 16 | 921600 |
| combined branch 3x3 | | 2764800 |

Observed fused pack time remains several milliseconds, which is far above what raw LPDDR bandwidth alone would imply for this byte count. The bottleneck is therefore likely small-panel address/control overhead, per-panel timing overhead during measurement, thread/chunk boundary effects, cache behavior, or mixed overhead rather than a simple sustained-bandwidth copy ceiling. This is diagnostic only; no exact bandwidth claim is made.
