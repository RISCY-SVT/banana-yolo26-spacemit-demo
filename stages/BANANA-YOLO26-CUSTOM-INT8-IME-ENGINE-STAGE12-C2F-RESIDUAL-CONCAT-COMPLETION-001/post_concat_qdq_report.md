# Post-Concat QDQ Report

tensor: `/model.2/Concat_output_0`
scale: `0.3288085460662842`
zero_point_u8: `2`
storage_handoff: `int8 signed storage = uint8_code - 128`
consumer: `/model.2/cv2/conv/Conv`

## Semantics

For each Concat float value:

```text
q = nearest_even(value / scale) + zero_point
q = clamp(q, 0, 255)
storage_s8 = int8(q - 128)
```

The selected path uses the same nearest-even utility family used by Stage 8/9
activation/requant paths.

## Correctness

Compact micro-ONNX oracle comparison:

- `concat_q_mismatches_against_ort=0`
- `concat_q_max_abs_diff_u8=0`

Board CPU0/1/2/3 Stage 12 fixtures:

- `concat_mismatches=0`
- `model2_cv2_mismatches=0`
