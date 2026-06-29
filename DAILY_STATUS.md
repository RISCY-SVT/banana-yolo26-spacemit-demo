# Daily Status 2026-06-29_21-43-36

## Goal For This Run

Minimize the YOLO26 INT8 Q/DQ rt204 SpaceMIT EP blocker and determine whether a
CPU-good, EP-good accelerated INT8 path exists without modifying the frozen
YOLO11 production repo.

## Done

- Verified the YOLO11 production repo remained at
  `production-2026-07-02 -> 9c0933be58ee122389d1a43f45f81e80655d6904`.
- Reconstructed prior FP32, Ultralytics Q/DQ, and manual ORT Q/DQ/QOperator
  candidates.
- Built synthetic minimal ONNX Conv/QDQ/Clip/QLinearConv repro models; all pass
  on rt204, so the blocker is not generic Q/DQ Conv or Clip parsing.
- Extracted the real YOLO26 first Conv/QDQ block and reproduced the rt204 EP
  compile failure at `/model.0/conv/Conv_token_1`.
- Verified provider pass filters do not fix the Q/DQ Conv compile blocker.
- Found the smallest correct Q/DQ fallback:
  `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv`.
- Tested QOperator e2e as a partial fallback: semantic smoke passes, but raw
  CPU/EP parity is loose, offload is not proven, and perf smoke is slower than
  FP32.

## Evidence

```text
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
```

## Open P0

None.

## Open P1

None for this R&D stage.

## Risks

- YOLO26 accelerated Q/DQ INT8 remains blocked by rt204 EP handling of the real
  YOLO26 Q/DQ Conv pattern.
- The correct provider-filter fallback is CPU-heavy and should not be presented
  as INT8 acceleration.
- QOperator needs a separate gate before any performance interpretation.

## Next Actions

- Package the extracted real-graph Q/DQ Conv repro for vendor/runtime feedback.
- Run a bounded QOperator fallback gate only if semantic parity/offload coverage
  can be tightened.
- Keep YOLO11 rt204 reevaluation separate from production policy.
