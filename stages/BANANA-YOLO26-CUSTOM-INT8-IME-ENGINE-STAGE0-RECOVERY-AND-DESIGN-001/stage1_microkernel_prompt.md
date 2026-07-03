# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001

You are Codex working in `/data/banana-yolo26-spacemit-demo` on branch
`yolo26-custom-int8-engine`.

Mission: implement and validate the first standalone `smt.vmadot` microkernel
for the YOLO26 custom INT8 engine skeleton. Do not implement the full engine.

Required scope:

- use accepted `smt.vmadot` 4x4x8 `s8xs8->s32` foundation;
- named asm preferred;
- raw `.insn` fallback only behind proof/disassembly;
- scalar oracle for exact int32 comparison;
- unit tests for packing, accumulator, accumulate/no-accumulate cases;
- microbench with scalar/RVV baseline if available;
- cluster0-only execution on CPU0-3;
- SIGILL guard in probe/test binaries;
- no CPU4-7 IME dispatch;
- no ncnn source mutation;
- no camera demo;
- no full COCO;
- no production FPS claim;
- no default backend change.

Key caveat from Stage 0:

The first real YOLO26 `MatMul` graph target is asymmetric activation x
activation quantization. Stage 1 should prove the pure signed microkernel
first; graph zero-point correction/requant integration belongs to a later
approved stage.
