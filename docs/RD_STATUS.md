# YOLO26 R&D Status

This repository is an isolated YOLO26 R&D workspace seeded from the frozen
YOLO11 production tag. It has no production claims and no active production
remote.

## Current Status

- Frozen YOLO11 production repo remains read-only and unchanged.
- Current accepted YOLO26 CPU oracle is Ultralytics `8.4.82`.
- Correct default/latest YOLO26 ONNX contract is end-to-end `[1,300,6]`.
- Traditional export is available with latest `end2end=False` and has shape
  `[1,84,8400]`.
- RT204 FP32 SpaceMIT EP smoke passes on BPI-F3/K1X for latest 640 end-to-end
  and traditional ONNX exports.
- Ultralytics `quantize=8` remains unsuitable: task-local small and
  representative calibration runs exported Q/DQ ONNX models, but the CPU oracle
  produced zero detections.
- Manual ONNX Runtime static Q/DQ over `Conv`/`MatMul` creates CPU-good INT8
  candidates for both YOLO26 end-to-end `[1,300,6]` and traditional
  `[1,84,8400]` contracts.
- RT204 SpaceMIT EP currently blocks board INT8: CPU-good Q/DQ candidates fail
  EP compilation with `output_type not implemented for clip minmax`.
- RT204 YOLO11 direct tensor-probe results are promising for a future separate
  adoption gate, but this R&D repository does not change the frozen YOLO11
  production policy.

## Raw Evidence

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
```

## Next Gate

Reduce or rewrite the INT8 Q/DQ pattern so rt204 can compile it, or obtain a
vendor-supported YOLO26 INT8 quantization recipe for SpacemiT ORT 2.0.4. Keep
YOLO11 rt204 reevaluation as a separate future adoption gate.
