# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001

## Mission

Use Stage31 evidence to choose the next Conv path without integrating the rejected direct/sliding sidecar.

Stage31 proved:

- `smt.vmadot1/2/3` replay remains valid.
- A real-node direct/sliding sidecar for `/model.4/m.0/cv1/conv/Conv` is byte-exact.
- The sidecar is slower than both single-thread and threaded MMT4D because panel-build dominates.

Stage32 must choose one:

1. Design a low-overhead sliding A-window layout that avoids the Stage31 `38901.3 us` panel-build cost, proof-only first.
2. Continue with MMT4D/threaded selected-cut work and do not spend more effort on direct/sliding until a no-panel-build design exists.
3. Stop custom Conv microkernel work temporarily and focus on broader full-model runner/value work.

## Hard boundaries

- No full engine unless explicitly authorized.
- No graph-wide scheduler.
- No production/model FPS claim.
- No `/data/ncnn` mutation.
- No CPU4-7 IME.
- No `vmadotn`.
- No default backend switch.

## Required gate

Any new direct/sliding design must first show a local panel-build reduction of at least 5x before attempting real-node integration.
