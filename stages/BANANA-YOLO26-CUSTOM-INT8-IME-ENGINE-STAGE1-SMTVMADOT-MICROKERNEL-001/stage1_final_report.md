# Stage 1 Final Report

classification: stage1-vmadot-microkernel-board-proven-ready-for-conv-stage
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
stage1_base_head_after_stage0_checkpoint: `d2ea43277cbf6873316527134c9931f9f8ee45df`
end_head: `d2ea43277cbf6873316527134c9931f9f8ee45df`
local_commit_stage1: false
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
microkernel: `smt.vmadot 4x4x8 s8xs8->s32`
asm_route: named
host_scalar_tests: pass
board_cluster0_tests: pass
microbench_done: yes
sliding_ops_note: created
stage2_prompt_created: true

## What changed

- Added `custom_int8_engine/include/y26_k1x_vmadot.h`.
- Added scalar `4x4x8 s8*s8->s32` oracle implementation.
- Added guarded RISC-V IME implementation using named `smt.vmadot`.
- Added host vector tests, board probe, deterministic vector generator, disassembly helper, board run helper, and microbench tool.
- Added Stage 1 reports under this stage directory.

## Proven

- Native host scalar-only build uses explicit `/usr/bin/g++`.
- Host CTest: 8/8 pass.
- RISC-V cross build with SpacemiT toolchain succeeds with named `smt.vmadot`.
- Disassembly contains `smt.vmadot v28,v0,v1`.
- Board CPU0-3 runs all deterministic vectors with status 0 and mismatches 0.
- Direct benchmark-only IME body on CPU0 is faster than scalar for this tiny 4x4x8 tile.

## Broken

- No full engine exists.
- No graph scheduler exists.
- No Conv integration exists.
- No zero-point correction is implemented in the microkernel.
- Public guarded API per-call timing is slower than scalar because it includes `sched_getcpu` and SIGILL handler setup per call.

## Unknown

- Full YOLO26 inference speed.
- Full-image pipeline speed.
- COCO/mAP accuracy.
- Conv1x1/Conv3x3 lowered-kernel performance after packing and zero-point compensation.

## Validation status

- `cmake -S custom_int8_engine -B .deps/custom_int8_engine/build-host-native ...`: pass
- `cmake --build .deps/custom_int8_engine/build-host-native`: pass
- `ctest --test-dir .deps/custom_int8_engine/build-host-native --output-on-failure`: pass
- RISC-V cross CMake/build under `.deps/custom_int8_engine/build-k1x-vmadot`: pass
- board probe CPU0-3: pass
- board microbench CPU0: pass
- final `git diff --check`: pass
- path-name hygiene for touched Stage 1 trees: pass, `path_issue_count=0`
- source hygiene scan for changed files, excluding command-log self-match: pass
- `find . -type l -print`: run and recorded

## Human decisions needed

- Approve Stage 2 before any Conv1x1/Conv3x3 lowering work.
- Decide whether the public guarded API should keep per-call SIGILL setup or move to a one-time runtime probe plus hot-path dispatch in a later runtime design.
