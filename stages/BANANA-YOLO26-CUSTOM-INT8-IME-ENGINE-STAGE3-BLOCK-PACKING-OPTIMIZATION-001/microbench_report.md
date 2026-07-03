# Microbench Report

All timings are board kernel/block evidence only. They are not YOLO26 inference FPS, full-image speed, camera speed, COCO mAP, or production readiness claims.

Board:

- Target: `svt@banana`
- Kernel: `Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64`
- Affinity: `taskset -c 0-3`
- Iterations: `3`

## Results

| case | shape | scalar us | old IME us | prepacked IME us | weight prepack us | activation/im2col us | correction us | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Conv1x1 | `160x160x32->32` | `109660` | `133531` | `46716.2` | `35.0143` | `33132.8` | `2683.75` | improved |
| Conv3x3 | `160x160x16->8` | `143646` | `390396` | `147974` | `39.334` | `157547` | `661.68` | packing-dominated |

## Interpretation

Prepacked Conv1x1 is materially better than scalar and the old Stage 2 wrapper. Conv3x3 correctness passes, but im2col/A packing remains the dominant cost and the prepacked path is not better than scalar for the measured real-node shape.
