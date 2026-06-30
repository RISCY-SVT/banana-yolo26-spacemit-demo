# Custom IME / Custom Operator Integration Feasibility

This is a read-only feasibility note. No IME kernels or custom ONNX Runtime ops
are implemented in this task.

## Possible Future Paths

| Path | Feasibility | Notes |
| --- | --- | --- |
| App-level custom kernels | Possible | Useful for preprocess/postprocess or standalone kernels outside ONNX Runtime graph execution. |
| ONNX Runtime custom ops | Possible | A shared library can register custom ops, but the ONNX graph must explicitly contain those ops. |
| Wrapper custom op around external runtime/model | Possible as R&D | Requires graph surgery and an ABI/memory contract. |
| Forked ORT or custom EP | Possible only as larger R&D | Requires maintaining provider integration and build/runtime compatibility. |

## Not a Near-Term Fix for the Current Blocker

The current blocker is inside closed rt204 SpaceMIT EP compilation of a Q/DQ
Conv subgraph:

```text
output_type not implemented for clip minmax
```

Replacing that Conv inside `libspacemit_ep.so` is not straightforward without
vendor cooperation. A custom ONNX Runtime op would not automatically patch the
same subgraph once it is owned by SpaceMIT EP. It would require explicit graph
rewriting so ORT dispatches a different custom node, and then a separate
performance/correctness gate.

## Evidence Required Before Implementation

- Compiler/assembler/disassembly/board execution proof for any IME kernel.
- Correctness oracle against FP32 and CPU-good INT8 outputs.
- ABI and memory-layout design for tensor inputs/outputs.
- ONNX graph surgery and custom-op registration proof.
- Performance comparison against rt204 FP32 and frozen YOLO11 production INT8.

## Decision

Custom IME is possible as a future experimental lane, but it is not a near-term
fix for the YOLO26 INT8 rt204 Q/DQ Conv compiler blocker.
