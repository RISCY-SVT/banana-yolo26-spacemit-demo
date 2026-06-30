# YOLO26 R&D Issue List

## P0

None.

## P1

None for the isolated R&D repo at this stage.

## P2

- YOLO26 full accelerated Q/DQ INT8 on rt204 is blocked by SpaceMIT EP
  compilation of Q/DQ Conv with explicit `kernel_shape`. The smallest repro is
  `15_conv_qdq_attr_kernel_shape.onnx`; the supplemental real-graph repro is
  `yolo26_first_conv_qdq_output_block.onnx`.
- Ultralytics `quantize=8` remains unsuitable for the tested YOLO26 checkpoint:
  exported Q/DQ models collapse CPU ORT detections to zero.
- QOperator e2e/traditional paths are not performance-gate ready: CPU/EP raw
  parity is loose, useful QLinear offload is not visible, bus semantics change
  under EP, and bounded timing smoke is slower than FP32.
- Stripped-kernel Q/DQ plus `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add` is
  a partial fallback candidate only. It restores smoke semantics but needs a
  separate placement/performance gate.
- YOLO11 rt204 adoption remains a separate future gate. The frozen production
  YOLO11 repository and `production-2026-07-02` policy are unchanged.
