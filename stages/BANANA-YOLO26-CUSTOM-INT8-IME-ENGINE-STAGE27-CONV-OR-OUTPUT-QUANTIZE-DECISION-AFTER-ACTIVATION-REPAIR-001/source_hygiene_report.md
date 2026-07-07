# Source Hygiene Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

## Checks

```text
git diff --check: pass
git diff --cached --check: pass
symlink scan custom_int8_engine stages: pass, no symlinks found
changed-file secret-like scan: pass, no findings
changed-file sensitive path scan: pass, no findings
```

## Build and Test

```text
host_native_configure: pass
host_native_build: pass
host_ctest: pass, 39/39
riscv_cross_configure: pass
riscv_cross_build: pass
board_stage26_replay: pass
board_threading_matrix: pass
```

## Notes

The RISC-V cross build emitted an existing warning in `bench_vmadot_microkernel.cpp` about `last_status` and `longjmp` clobbering. Stage27 did not modify that file.

No `/data/ncnn` mutation was performed. No full engine, graph expansion, `vmadot1/2/3`, `vmadotn`, FP/vfmadot, CPU4-7 IME, OpenMP/all-core dispatch, COCO/mAP, camera/full-image, model FPS, production, or default-backend claim was made.
