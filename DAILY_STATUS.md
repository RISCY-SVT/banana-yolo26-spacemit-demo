# Daily Status 2026-06-30_06-12-26

## Goal For This Run

Create a vendor-grade rt204 Q/DQ Conv repro, gate QOperator, and decide whether
any YOLO26 INT8 fallback is ready for a next performance stage without
modifying the frozen YOLO11 production repo.

## Done

- Verified the YOLO11 production repo remained at
  `production-2026-07-02 -> 9c0933be58ee122389d1a43f45f81e80655d6904`.
- Reconstructed prior FP32, Ultralytics Q/DQ, and manual ORT Q/DQ/QOperator
  candidates.
- Built and tested a minimal repro family on CPU and rt204 EP.
- Reduced the first blocker to a tiny synthetic Q/DQ Conv model with explicit
  `kernel_shape=[3,3]`: `15_conv_qdq_attr_kernel_shape.onnx`.
- Kept `yolo26_first_conv_qdq_output_block.onnx` as the supplemental real YOLO26
  repro.
- Gated QOperator and rejected it for performance benchmarking because useful
  QLinear offload is not visible and CPU/EP semantics are loose.
- Found a CPU-exact model rewrite that strips optional Conv `kernel_shape`
  attributes. It avoids the first Conv blocker, but exposes an attention MatMul
  issue.
- Found a partial fallback candidate for the stripped model:
  `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add`.

## Evidence

```text
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
```

## Open P0

None.

## Open P1

None for this R&D stage.

## Risks

- YOLO26 full accelerated Q/DQ INT8 remains blocked by rt204 EP handling of
  Q/DQ Conv with explicit `kernel_shape`.
- The stripped-kernel `MatMul;Add` fallback is only a candidate and needs a
  separate placement/performance gate.
- QOperator should not be used as a performance path without future runtime
  evidence of QLinear offload and stable semantics.

## Next Actions

- Send the tiny synthetic repro plus the real YOLO26 repro to the runtime
  vendor.
- Run a narrow partial-fallback placement/performance gate for the
  stripped-kernel Q/DQ model only if useful offload can be proven.
- Keep YOLO11 rt204 reevaluation separate from production policy.
