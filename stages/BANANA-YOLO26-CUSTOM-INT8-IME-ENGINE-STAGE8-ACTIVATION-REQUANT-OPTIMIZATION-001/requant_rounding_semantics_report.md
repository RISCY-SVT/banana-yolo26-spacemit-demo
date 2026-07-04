# Requant Rounding Semantics Report

classification: pass

## Current Reference Semantics

The existing Stage 7 oracle path and C++ reference use `std::nearbyint`, which follows the active floating-point environment. In the current container and board runs this behaved as nearest-even for the controlled edge cases.

Current quantization helper:

```text
scaled = value / scale
q = nearbyint(scaled) + zero_point
q = clamp(q, 0, 255)
```

## Boundary Metadata

| boundary | scale | zero_point_u8 | signed storage |
|---|---:|---:|---:|
| images Q | 0.00392156885937 | 0 | -128 |
| Conv0 Q | 0.620968401432 | 128 | 0 |
| Act0 Q | 0.311162889004 | 1 | -127 |
| Conv1 Q | 1.1142437458 | 122 | -6 |
| Act1 Q | 0.582500994205 | 0 | -128 |
| Conv2 input | 0.582500994205 | 0 | -128 |

## Fixed-Point Diagnostic

`test_requant_fixed_point` covers nearest-even ties, signed values, and clamp behavior for exact multipliers such as `0.5` and `0.25`. The selected Stage 8 runtime path remains `int8_lut`, not fixed-only, because exact fixture matching is required.
