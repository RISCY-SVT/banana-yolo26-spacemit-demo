# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE18-CLUSTER0-THREADED-CONV-INTEGRATION-001

Implement a bounded cluster0 threaded Conv sidecar for the representative `/model.4` branch-entry Conv path.

Use Stage17 evidence:

```text
threading_feasibility: strong_positive
4-thread speedup: 3.680290x
partition: spatial row split
CPU policy: CPU0-3 only
mismatches: 0
```

Scope:

```text
- integrate one non-default threaded Conv runner path for /model.4/m.0/cv1/conv/Conv
- keep scalar and single-thread IME paths
- keep thread count explicit: 1, 2, 3, 4
- no OpenMP/all-core default
- no CPU4-7 IME
- no graph-wide scheduler
- no full YOLO26 inference
- no camera/full-image/COCO/mAP
- no model FPS or production claim
```

Acceptance:

```text
host CTest pass
RISC-V cross build pass
board CPU0-3 correctness pass
stable mean/stddev benchmark
4-thread path remains >=1.5x faster than single-thread on representative/full-shape branch-entry
```
