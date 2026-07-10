# Suffix Quant Boundary Report

All cumulative suffix cuts use:

```text
input_boundary: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
input_dtype: uint8
input_shape: 1x128x80x80
```

The provisional Stage42 target `model.16` has:

```text
block_id: model.16
first_node: /model.16/cv1/conv/Conv
last_node: /model.16/cv2/act/Mul_output_0_DequantizeLinear
node_count: 66
operator_mix: Add:2, Concat:2, Conv:9, DequantizeLinear:17, Mul:9, QuantizeLinear:17, Sigmoid:9, Split:1
output_boundary: /model.16/cv2/act/Mul_output_0_DequantizeLinear_Output
output_shape: 1x64x80x80
cumulative_cut_status: pass
```

The block has a quantized internal structure, but its current selected cumulative output boundary is dequantized. Stage42 must decide whether to compare at this dequantized output or choose an earlier quantized boundary that preserves exactness and avoids unnecessary float round-trip.
