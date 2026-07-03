# Performance Gate Plan

No production FPS claim is authorized in Stage 0.

## Gates

- Gate 0: build + no SIGILL + oracle pass
- Gate 1: one `smt.vmadot` microkernel correct and faster than scalar/RVV reference
- Gate 2: Conv1x1/3x3 kernel faster than scalar/RVV, packing cost separated
- Gate 3: first graph block faster than reference for same tensor contract
- Gate 4: full forward faster than YOLO26 FP16 keep-I/O reference or clear blocker
- Gate 5: full forward faster than old YOLO11 dynamic640 INT8 production reference if comparable
- Gate 6: full image/full pipeline improves too
- Stretch: 20+ FPS aspirational, not Stage 0 or Stage 1 acceptance

## Benchmark Protocol

```text
--pin cluster0
--threads 4
--warmup 10
--runs 100
--repeats 5
mean/stddev
exact command
model/image/output hashes
```

Packing time must be reported separately from kernel time.
