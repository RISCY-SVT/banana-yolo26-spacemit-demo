# Concat Layout Plan

selected_layout: `NHWC`
selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`

## Channel Order

For each spatial position:

```text
channels 0..15   = /model.2/Split_output_0
channels 16..31  = /model.2/Split_output_1_DequantizeLinear_Output
channels 32..47  = /model.2/m.0/Add_output_0
```

## Stage 12 Path

1. Materialize `concat_f32` in the channel order above.
2. Quantize `concat_f32` with `/model.2/Concat_output_0_scale` and zero-point.
3. Store as signed int8 storage (`uint8_code - 128`) for `/model.2/cv2/conv/Conv`.

Direct view/memcpy-only Concat is not accepted because inputs are mixed float
domains and post-Concat Q/DQ must be applied after merge.

Future optimization: fuse Concat layout and QuantizeLinear to avoid materialized
`concat_f32`, but only with the same oracle.
