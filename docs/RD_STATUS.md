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
- RT204 SpaceMIT EP currently blocks accelerated board Q/DQ INT8: CPU-good Q/DQ
  candidates fail EP compilation with `output_type not implemented for clip
  minmax`. The minimized failing model is the extracted real YOLO26 first Conv
  Q/DQ block, not a synthetic toy Conv/QDQ/Clip graph.
- Provider pass filters do not avoid the compile failure. The smallest correct
  Q/DQ fallback is
  `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv`,
  which is CPU-heavy and not an accelerated INT8 path.
- A QOperator e2e Conv+MatMul candidate runs semantically under rt204, but raw
  CPU/EP parity is loose, offload is not proven for `QLinearConv`/`QLinearMatMul`,
  and perf smoke is slower than FP32. It needs a separate fallback gate.
- RT204 YOLO11 direct tensor-probe results are promising for a future separate
  adoption gate, but this R&D repository does not change the frozen YOLO11
  production policy.

## Raw Evidence

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
```

## Next Gate

Package the extracted real YOLO26 Q/DQ Conv repro for vendor/runtime feedback,
then run a separate QOperator fallback gate only if semantic parity and offload
coverage can be tightened. Keep YOLO11 rt204 reevaluation as a separate future
adoption gate.
