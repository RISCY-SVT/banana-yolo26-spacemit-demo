# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE3-BLOCK-PACKING-OPTIMIZATION-001

Implement the next narrow stage for the custom YOLO26n INT8 IME backend.

## Mission

Optimize the Stage 2 Conv1x1/Conv3x3 MMT4D paths at block level without implementing full YOLO26 inference.

## Required Inputs

- Stage 0 graph and quantization reports.
- Stage 1 `smt.vmadot` board-proven microkernel.
- Stage 2 cached runtime probe, Conv1x1/Conv3x3 correctness fixtures, and microbench reports.

## Scope

- Select one real Conv or Conv-like graph block from the accepted Q/DQ ONNX metadata.
- Add B prepacking for Conv weights.
- Add A block reuse or tiled im2col buffering.
- Keep raw signed dot kernels separate from zero-point correction and requantization.
- Add row/column sum support only for the selected block if needed.
- Compare against scalar and ONNX CPU oracle dumps for that block.
- Benchmark kernel-only and packing-included block paths on CPU0-3.

## Non-Goals

- No full YOLO26 engine.
- No graph scheduler.
- No ncnn source mutation.
- No camera demo.
- No COCO/mAP.
- No model FPS or production claim.
- No `vmadot1/2/3`, `vmadotn`, or FP/`vfmadot` implementation.

## Acceptance

- Host tests pass.
- Cross build uses named `smt.vmadot`.
- Board CPU0-3 correctness passes for selected block.
- Packing cost is separated from core tile timing.
- Report explicitly states whether the block is faster than scalar after packing.
