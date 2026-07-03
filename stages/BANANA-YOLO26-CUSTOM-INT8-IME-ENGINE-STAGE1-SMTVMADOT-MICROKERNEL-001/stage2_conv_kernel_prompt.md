# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE2-CONV1X1-3X3-KERNEL-BRINGUP-001

You are Codex working in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine`.

Use Stage 1 as authority for the first IME primitive:

- primitive: `smt.vmadot`
- semantics: `4x4x8 s8xs8->s32`
- A layout: `4x8` row-major, `A[m*8+k]`
- B layout: `4x8` transposed output-major, `B[n*8+k]`
- C layout: `4x4` row-major, `C[m*4+n]`
- CPU policy: cluster0 CPU0-3 only
- no CPU4-7 IME dispatch
- no `vmadot1`, `vmadot2`, `vmadot3`, `vmadotn`, FP, or vfmadot implementation
- no full engine
- no ncnn source mutation

Stage 2 mission:

1. Implement a narrow packed GEMM/MMT4D tile loop around the proven Stage 1 microkernel.
2. Bring up one real graph block or synthetic Conv1x1/Conv3x3 lowering target selected from Stage 0.
3. Keep zero-point correction outside the Stage 1 microkernel and document compensation.
4. Compare scalar/RVV/lowered IME outputs against deterministic fixtures.
5. Benchmark packing cost separately from hot kernel cost.
6. Do not implement scheduler, full YOLO26 inference, camera demo, full COCO, or production artifacts.

Required proof:

- host scalar correctness
- RISC-V cross build
- disassembly still shows `smt.vmadot`
- board cluster0 correctness on CPU0-3
- microbench with packing separated
- no production FPS or model-level performance claims
