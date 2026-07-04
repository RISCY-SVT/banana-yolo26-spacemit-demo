# Activation LUT Reference Semantics

classification: pass

Stage 8 LUTs are per-boundary 256-entry tables:

```text
uint8 conv_out_code -> int8 act_out_signed
```

Reference formula:

```text
x = conv_output_scale * (conv_out_code - conv_output_zero_point)
y = x / (1 + exp(-x))
act_code = nearest_even(y / act_output_scale) + act_output_zero_point
act_code = clamp(act_code, 0, 255)
act_signed = int8(act_code - 128)
```

The LUT does not change Conv int32 correction semantics. The runtime still computes the corrected Conv accumulator first, quantizes it to the Conv output code, then applies the per-boundary LUT.

Implemented files:

- `custom_int8_engine/include/y26_k1x_activation.h`
- `custom_int8_engine/kernels/activation_requant.cpp`
- `custom_int8_engine/tests/test_stage8_activation_requant.cpp`
