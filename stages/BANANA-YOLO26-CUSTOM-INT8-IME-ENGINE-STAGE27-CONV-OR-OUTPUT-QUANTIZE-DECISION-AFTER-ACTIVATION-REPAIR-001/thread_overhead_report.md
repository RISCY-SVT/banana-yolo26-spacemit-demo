# Thread Overhead Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

## Current Implementation

Source inspection shows `Y26ThreadedConvWorkspace` creates worker threads during prepare and reuses them through barriers:

```text
custom_int8_engine/kernels/conv_threaded.cpp
```

The Stage27 `thread_overhead_us` bucket therefore does not represent per-inference thread creation/join. It is the measured gap between total threaded call time and max worker time, including barrier, copy/writeback, scheduling, and control overhead inside already persistent per-node worker workspaces.

## Measured Overhead

Stage26 replay:

```text
thread_overhead_us: 4980.87
total_us: 41669.2
thread_overhead_share_of_total_pct: 11.9558
```

Current all-4 matrix replay:

```text
thread_overhead_us: 4710.67
total_us: 41642.8
thread_overhead_share_of_total_pct: 11.3121
```

## Decision

`SELECT_C2_PERSISTENT_POOL` is not selected in Stage27 because the current implementation already has persistent per-node workers. A cross-node shared worker region could be a future task, but it is not a cheap local Stage27 win and would require new scheduling/barrier ownership across heterogeneous Conv nodes.
