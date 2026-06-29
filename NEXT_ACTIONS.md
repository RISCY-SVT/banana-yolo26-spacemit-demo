# Next Actions

## Immediate YOLO26 R&D Gate

- Package the extracted `yolo26_first_conv_qdq_output_block.onnx` repro and
  supporting logs for vendor/runtime feedback. This is the smallest real-graph
  failure found so far.
- Ask the runtime vendor whether SpacemiT ORT `2.0.4` supports the real YOLO26
  ONNX Runtime static Q/DQ Conv pattern and whether the internal `clip minmax`
  output-type failure has a provider option or model-rewrite workaround.
- Run a separate QOperator fallback gate only if needed. It must tighten
  CPU/EP parity, prove or disprove `QLinearConv`/`QLinearMatMul` offload, and
  explain why smoke latency is slower than FP32.

## YOLO11 RT204 Follow-Up

- Keep YOLO11 production frozen on the `production-2026-07-02` policy.
- If rt204 adoption becomes desirable, open a separate YOLO11 gate that reruns
  the full app, loader, camera, and performance matrix against rt204. The direct
  tensor-probe signal from this R&D pass is not enough for adoption.

## Preserve

- Do not modify `/data/banana-yolo11-spacemit-demo` from this R&D repo.
- Do not claim YOLO26 production readiness.
- Do not attempt INT8 performance claims until SpaceMIT EP executes a
  CPU-good INT8 candidate with proven useful offload.
