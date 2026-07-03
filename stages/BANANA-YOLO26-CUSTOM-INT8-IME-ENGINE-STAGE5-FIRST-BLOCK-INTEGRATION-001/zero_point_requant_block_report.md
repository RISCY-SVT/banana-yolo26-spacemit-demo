# Zero-Point And Requant Block Report

Stage 5 implements the zero-point correction boundary for the selected first Conv block only.

Raw IME kernel computes:

```text
raw = sum((q_u8 - 128) * w_s8)
```

The corrected int32 block output is:

```text
corrected = raw + (128 - activation_zero_point_u8) * sum(w_s8) + bias_i32
```

This is equivalent to:

```text
sum((q_u8 - za) * (w_s8 - zw)) + bias_i32
```

for the selected node because:

- activation zero-point `za = 0`
- weight zero-points `zw = 0` for all output channels
- engine input storage is `q_u8 - 128`
- padding uses `input_storage_zero_point_s8 = -128`

The selected Conv output Q/DQ metadata is recorded:

- output scale: `0.620968401432`
- output zero-point: `128`

Stage 5 does not integrate downstream output requant, SiLU, or full graph activation policy. The selected block runner returns corrected int32 output for exact oracle comparison.
