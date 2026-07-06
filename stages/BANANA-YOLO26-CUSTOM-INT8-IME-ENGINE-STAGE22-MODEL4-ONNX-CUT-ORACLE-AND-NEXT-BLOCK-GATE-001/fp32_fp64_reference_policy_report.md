# FP32/FP64 Reference Policy Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## Policy

The Stage22 authority for the `/model.4` C2f cut is ONNX Runtime CPU behavior on the accepted Q/DQ ONNX graph.

```text
authority: ONNX Runtime CPUExecutionProvider
model_dtype_policy: original Q/DQ ONNX graph semantics
accepted_output_boundary: uint8 QDQ code
cpp_compare_boundary: uint8 QDQ code after NHWC conversion
```

No fp64 scalar reference is used as final truth for this stage. C++ scalar/reference paths are diagnostic and must match the ONNX cut output at the final quantized boundary.

## Rounding

Stage22 uses explicit nearest-even quantization helpers and a scoped RNE guard for the same-input verifier path so ambient RISC-V `frm` does not alter the accepted ONNX-equivalent result. The board regression sweep across RNE/RTZ/RDN/RUP/RMM passed with `mismatches=0` and restored the caller's ambient `frm`.
