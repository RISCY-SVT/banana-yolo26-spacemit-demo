# Activation/Requant Boundary Plan

Selected non-Conv boundary: `/model.0/conv/Conv_output_0` Q/DQ followed by SiLU represented as `/model.0/act/Sigmoid` + `/model.0/act/Mul`, followed by `/model.0/act/Mul_output_0` Q/DQ.

## Stage 6 Policy

- Correctness first.
- Activation/requant is implemented as scalar float fallback.
- Activation/requant timing is measured separately as `activation_us`.
- Conv nodes use existing plain `smt.vmadot` MMT4D IME kernels where board build enables IME.
- No LUT or approximated integer SiLU is claimed in Stage 6.

## Formula

For each Conv0 output element:

```text
conv0_float[oc] = conv0_i32[oc] * images_scale * conv0_weight_scale[oc]
conv0_q = clamp_uint8(round_to_nearest_even(conv0_float / conv0_output_scale) + conv0_output_zero_point)
conv0_dq = (conv0_q - conv0_output_zero_point) * conv0_output_scale
silu = conv0_dq / (1 + exp(-conv0_dq))
act_q = clamp_uint8(round_to_nearest_even(silu / act0_output_scale) + act0_output_zero_point)
conv1_input_s8 = int8(act_q - 128)
```

Rounding policy: nearest-even to match ONNX `QuantizeLinear` behavior used by the CPU oracle.

Clamp range: `[0,255]` before uint8 cast.

## Accuracy Oracle

The Python oracle extracts ONNX Runtime CPU outputs for the same selected nodes and also generates small deterministic local fixtures. Exact C++ tests compare selected subset corrected int32 output against the generated oracle fixture.

## Timing Bucket

- `conv0_*`: Conv0 IME/scalar work and correction.
- `activation_us`: Conv0 Q/DQ + SiLU + act Q/DQ fallback.
- `conv1_*`: Conv1 IME/scalar work and correction.

