# Daily Status 2026-06-29_16-56-34

## Goal For This Run

Investigate YOLO26 INT8 calibration/export behavior and SpacemiT ORT `2.0.4`
operator support without modifying the frozen YOLO11 production repo.

## Done

- Verified the YOLO11 production repo remained at
  `production-2026-07-02 -> 9c0933be58ee122389d1a43f45f81e80655d6904`.
- Built task-local oracle and calibration image sets under the YOLO26 repo.
- Reconfirmed YOLO26 float CPU oracle and rt204 FP32 SpaceMIT EP parity.
- Proved Ultralytics `quantize=8` Q/DQ exports produce zero detections in CPU
  ORT for both end-to-end and traditional YOLO26 contracts.
- Produced manual ONNX Runtime static Q/DQ INT8 candidates that are CPU-good.
- Proved rt204 SpaceMIT EP currently fails to compile those CPU-good Q/DQ
  candidates with `output_type not implemented for clip minmax`.
- Ran provider filter diagnostics and a diagnostic perf smoke.
- Rechecked frozen YOLO11 models through rt204 direct tensor probe as R&D-only
  evidence.

## Evidence

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
```

## Open P0

None.

## Open P1

None for this R&D stage.

## Risks

- YOLO26 INT8 acceleration is blocked by rt204 EP Q/DQ/Conv compile support.
- Manual CPU fallback can recover correctness but is too slow and not an
  accelerated board INT8 solution.

## Next Actions

- Reduce the Q/DQ failure to a minimal model or obtain a vendor-supported rt204
  YOLO26 INT8 quantization recipe.
- Keep YOLO11 rt204 reevaluation separate from production policy.
