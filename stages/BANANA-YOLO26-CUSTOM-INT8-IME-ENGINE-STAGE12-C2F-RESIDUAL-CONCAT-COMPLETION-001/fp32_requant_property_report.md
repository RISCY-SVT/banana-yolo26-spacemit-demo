# FP32 Requant Property Report

scope: Stage 12 Add/Concat/post-Concat QDQ

## Checked Properties

- Add is evaluated in float domain, matching the accepted ONNX graph boundary.
- Concat is evaluated in float domain before QuantizeLinear.
- Post-Concat QDQ uses nearest-even rounding, clamp to `[0,255]`, and signed storage shift.
- Compact micro-ONNX `Add -> Concat -> QuantizeLinear` oracle was generated and compared.

## Result

- `concat_q_mismatches_against_ort=0`
- `concat_q_max_abs_diff_u8=0`

## Caveat

This report does not claim a generic ONNX runtime implementation. It validates
only the selected Stage 12 C2f boundary and its deterministic compact fixtures.
