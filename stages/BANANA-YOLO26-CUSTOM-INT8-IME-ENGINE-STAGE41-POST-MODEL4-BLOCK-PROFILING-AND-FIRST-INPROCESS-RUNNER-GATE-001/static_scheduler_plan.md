# Static Scheduler Plan

Stage41 does not implement a graph-wide scheduler.

Future static scheduler requirements:

```text
topological block list from fixed YOLO26 graph
explicit tensor lifetimes
explicit CPU affinity for each block
no hardware_concurrency default
no OpenMP/all-core dispatch
IME blocks pinned to CPU0-3 only
fallback/debug ORT cuts outside final fast path
```

The scheduler should be introduced only after at least one post-model4 custom block is correct in the in-process scaffold.
