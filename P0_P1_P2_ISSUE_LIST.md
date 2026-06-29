# YOLO26 R&D Issue List

## P0

None.

## P1

None for the isolated R&D repo at this stage.

## P2

- YOLO26 accelerated Q/DQ INT8 on rt204 is blocked by SpaceMIT EP compilation
  of the real YOLO26 Q/DQ Conv pattern. The minimized repro is
  `yolo26_first_conv_qdq_output_block.onnx`, which fails at
  `/model.0/conv/Conv_token_1` with
  `output_type not implemented for clip minmax`.
- Ultralytics `quantize=8` remains unsuitable for the tested YOLO26 checkpoint:
  exported Q/DQ models collapse CPU ORT detections to zero.
- QOperator e2e Conv+MatMul is a partial fallback candidate only. It gives sane
  semantics in smoke tests, but CPU/EP raw parity is loose, useful offload is
  not proven, and perf smoke is slower than FP32.
- YOLO11 rt204 adoption remains a separate future gate. The frozen production
  YOLO11 repository and `production-2026-07-02` policy are unchanged.
