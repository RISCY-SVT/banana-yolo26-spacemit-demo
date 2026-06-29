# Next Actions

## Immediate YOLO26 R&D Gate

- Reduce the CPU-good manual ONNX Runtime Q/DQ graph to the smallest failing
  rt204 EP compile case around the first Conv/Q/DQ region.
- Test whether an alternate static quantization format, quantizing fewer op
  types, or excluding the first stem Conv avoids `output_type not implemented
  for clip minmax` without losing CPU oracle correctness.
- Ask the runtime vendor whether SpacemiT ORT `2.0.4` supports the ONNX Runtime
  static Q/DQ Conv/MatMul pattern used by YOLO26, and whether a different
  quantization recipe is required.

## YOLO11 RT204 Follow-Up

- Keep YOLO11 production frozen on the `production-2026-07-02` policy.
- If rt204 adoption becomes desirable, open a separate YOLO11 gate that reruns
  the full app, loader, camera, and performance matrix against rt204. The direct
  tensor-probe signal from this R&D pass is not enough for adoption.

## Preserve

- Do not modify `/data/banana-yolo11-spacemit-demo` from this R&D repo.
- Do not claim YOLO26 production readiness.
- Do not attempt INT8 performance claims until SpaceMIT EP executes a
  CPU-good INT8 candidate.
