# YOLO26 K1X INT8 Executor

This repository contains a standalone experimental YOLO26n-640 executor for
Banana-Pi BPI-F3 / SpacemiT K1X. It implements the frozen
`K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` profile with exact Q62 integer
arithmetic, resident `NCHWc8_SPATIAL_INNER_V1` feature tensors, and explicit
SpacemiT IME kernels.

The runtime does not load ONNX Runtime or Python. Its measured graph path has no
per-run allocation, per-run file I/O, float Q/DQ materialization, or string
operator dispatch. The model package is prepared once and execution uses a
static numeric operation schedule.

The safe default is four IME workers pinned to CPU0-3, a controller on CPU4,
and `SCHED_OTHER`. CPU4-7 never execute IME instructions. `rr20` is a bounded
lab mode, not the deployment default.

The Stage52 handoff was tested on Bianbu 2.2.1 with Linux 6.6.63
`#2.2.7.2`, SpacemiT GCC 14.3.0, and board NVMe `/data`. Its frozen identities
are:

```text
model SHA-256:
  30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
package asset_hashes.tsv SHA-256:
  d3b4cb794f1373aa712d77bab177a5f7da58530361c9af58c0caf5bbcd6dc75f
```

Start with:

- [BUILDING_K1X_INT8_EXECUTOR.md](BUILDING_K1X_INT8_EXECUTOR.md)
- [DEPLOYING_K1X_INT8_EXECUTOR.md](DEPLOYING_K1X_INT8_EXECUTOR.md)
- [K1X_INT8_EXECUTOR_API.md](K1X_INT8_EXECUTOR_API.md)
- [K1X_INT8_EXECUTOR_LIMITATIONS.md](K1X_INT8_EXECUTOR_LIMITATIONS.md)

This remains an experimental K1X-specific executor. It is not a production or
camera-service release, and no 20 FPS claim is made.
