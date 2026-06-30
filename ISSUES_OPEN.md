# Open Issues

## P0

None for the isolated YOLO26 R&D repo.

## P1

None for the current forensic stage.

## P2

- YOLO26 INT8 board EP: full CPU-good manual Q/DQ candidates fail rt204
  SpaceMIT EP compilation with `output_type not implemented for clip minmax`.
  The smallest repro is `15_conv_qdq_attr_kernel_shape.onnx`; the first YOLO26
  Conv/QDQ block remains the supplemental real-graph repro.
- Ultralytics `quantize=8` preset: Q/DQ exports score-collapse to zero
  detections in CPU ORT.
- QOperator fallback: not performance-gate ready because useful QLinear offload
  is not visible and CPU/EP semantics are loose.
- Stripped-kernel Q/DQ with `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add`:
  partial fallback candidate only; requires a separate placement/performance
  gate.
- YOLO11 rt204 adoption: direct tensor-probe smoke is positive, but full
  production app/camera/perf regression is still required in a separate task.
