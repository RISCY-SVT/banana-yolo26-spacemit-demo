# Stage 4 Microbench Report

All timings are selected kernel/block evidence only. They are not YOLO26 inference FPS, full-image speed, camera speed, COCO mAP, or production readiness claims.

Board:

- Target: `svt@banana`
- Kernel: `Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64`
- Correctness affinity: `taskset -c 0-3`
- Benchmark affinity: `taskset -c 0`
- Iterations: `5`

| case | scalar us | old wrapper us | Stage 3 prepacked us | Stage 4 M-major us | Stage 4 N-major us | packA probe us | correction us | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Conv1x1 `160x160x32->32` | `112047` | `137374` | `46649.6` | `21843.2` | `65855.4` | `6260.54` | `2649.96` | repaired |
| Conv3x3 `160x160x16->8` | `147333` | `391010` | `149121` | `37097.9` | `65852.3` | `24951.5` | `658.726` | repaired |

M-major is selected. N-major is correct but slower for the selected shapes.
