# Next Actions

## Immediate YOLO26 R&D Gate

- Send the tiny vendor repro `15_conv_qdq_attr_kernel_shape.onnx` and the
  supplemental `yolo26_first_conv_qdq_output_block.onnx` to the runtime vendor.
  Ask whether SpacemiT ORT `2.0.4` supports Q/DQ Conv with explicit
  `kernel_shape` and whether the internal `clip minmax` output-type failure has
  a provider option or patch.
- If continuing locally, run a narrow partial-fallback placement/performance
  gate for the stripped-kernel Q/DQ model with
  `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add`. That gate must prove useful
  EP placement and compare against FP32 without making production claims.
- Do not pursue QOperator for performance unless a future runtime can show
  `QLinearConv`/`QLinearMatMul` offload and stable CPU/EP semantics.

## YOLO11 RT204 Follow-Up

- Keep YOLO11 production frozen on the `production-2026-07-02` policy.
- If rt204 adoption becomes desirable, open a separate YOLO11 gate that reruns
  the full app, loader, camera, and performance matrix against rt204. The direct
  tensor-probe signal from this R&D pass is not enough for adoption.

## Preserve

- Do not modify `/data/banana-yolo11-spacemit-demo` from this R&D repo.
- Do not claim YOLO26 production readiness.
- Do not attempt INT8 performance claims until SpaceMIT EP executes a CPU-good
  INT8 candidate with proven useful offload.
