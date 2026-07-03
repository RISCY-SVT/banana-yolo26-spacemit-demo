# Stage 2 Final Report

classification: stage2-conv-kernels-board-proven-ready-for-block-stage
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE2-CONV1X1-3X3-KERNEL-BRINGUP-001
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
initial_head: `d2ea43277cbf6873316527134c9931f9f8ee45df`
stage1_checkpoint_head: `584db270df25cb4272d424f6846362e74b5e83e1`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
runtime_probe_hotpath: pass
conv1x1_status: pass
conv3x3_status: pass
zero_point_correction_integrated: synthetic-only
sliding_ops_note: updated
host_tests: pass
board_tests: pass
microbench_done: yes

## What changed

- Added cached IME runtime probe and cluster0 hotpath API.
- Added unsafe cluster0 `smt.vmadot` tile call for already-pinned internal loops.
- Added Conv1x1 scalar and IME paths.
- Added Conv3x3 scalar and im2col/MMT4D IME paths.
- Added MMT4D pack helpers with tail zero-padding.
- Added runtime probe, Conv, packing, and zero-point formula tests.
- Added hotpath and Conv kernel microbench tools.
- Added Stage 2 reports and Stage 3 prompt.

## Proven

- Host CTest: 14/14 pass.
- RISC-V cross build: pass.
- Disassembly contains named `smt.vmadot`; raw `.insn` was not used.
- Board probe succeeds on CPU0/1/2/3.
- Board direct `smt.vmadot` fixture passes on CPU0/1/2/3 with mismatches 0.
- Board Conv1x1 fixtures pass with mismatches 0.
- Board Conv3x3 fixtures pass with mismatches 0.
- Microkernel public cached wrapper is faster than scalar in no-packing tile benchmark.
- MMT4D tile core is faster than scalar for Conv-derived tile benchmarks.

## Broken

- Current Conv wrappers are not performance-ready when packing/im2col is included.
- Full engine, graph scheduler, real graph block integration, requantization, and activation functions remain unimplemented.

## Unknown

- Full YOLO26 inference speed.
- Full-image pipeline speed.
- COCO/mAP.
- Real graph accuracy after zero-point correction and requantization.

## Validation Commands

See `commands.txt` and `$LOG_DIR/run_logs/` for exact commands and exit codes.

## Human Decisions Needed

- Approve Stage 3 block/packing optimization.
- Decide whether Stage 3 may use an approved Python `onnx` environment to select the first real Conv block.
