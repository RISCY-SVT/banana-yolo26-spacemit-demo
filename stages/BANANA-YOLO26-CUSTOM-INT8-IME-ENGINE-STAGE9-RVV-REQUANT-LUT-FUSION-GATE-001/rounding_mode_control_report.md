# Rounding Mode Control Report

classification: pass

Stage 8 reference semantics are preserved for accepted Stage 9 paths.

Reference:

```text
conv_float = float(int32_accumulator) * float(input_scale * weight_scale[channel])
conv_code = nearest_even(conv_float / conv_output_scale) + conv_output_zero_point
conv_code = clamp(conv_code, 0, 255)
activation_signed = silu_lut[conv_code]
```

Checks:

- Stage 8 LUT internal reference: pass.
- Standalone ONNX Runtime Q/DQ+SiLU 256-code LUT oracle: pass for Act0 and Act1.
- A1 scalar-unrolled LUT: exact vs Stage 8 LUT for selected fixtures.
- A2 RVV f32 LUT: exact vs Stage 8 LUT for selected fixtures and full selected-subset runner on CPU0/1/2/3.
- A3 fixed-requant LUT: exact vs Stage 8 LUT for selected fixtures and full selected-subset runner on CPU0/1/2/3.

Caveat: A2 uses RVV f32 conversion/multiply/divide/convert with the board's current rounding behavior. It is accepted only for the selected Stage 7/8 subset evidence in this stage, not as a global all-tensor quantization proof.
