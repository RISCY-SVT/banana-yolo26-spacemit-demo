# Open Issues

## P0

None for the isolated YOLO26 R&D repo.

## P1

None for the current forensic stage.

## P2

- YOLO26 INT8 board EP: CPU-good manual Q/DQ candidates fail rt204 SpaceMIT EP
  compilation with `output_type not implemented for clip minmax`. The minimized
  real-graph repro is the first YOLO26 Conv/QDQ block.
- Ultralytics `quantize=8` preset: Q/DQ exports score-collapse to zero
  detections in CPU ORT.
- QOperator e2e fallback: semantically promising but not proven as accelerated
  offload and slower than FP32 in smoke testing.
- YOLO11 rt204 adoption: direct tensor-probe smoke is positive, but full
  production app/camera/perf regression is still required in a separate task.
