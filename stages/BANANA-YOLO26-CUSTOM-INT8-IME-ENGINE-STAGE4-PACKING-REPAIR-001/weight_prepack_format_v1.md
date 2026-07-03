# Weight Prepack Format v1

Object: `Y26PrepackedConvWeights`

The public type is opaque. The implementation stores:

- original `Y26Conv2DParams`
- `kernel_h`
- `kernel_w`
- `kernel_k = kernel_h * kernel_w * input_c`
- `packed_b_bytes`
- `total_bytes`
- `source_tensor_name`
- `quant_scale_metadata`
- aligned packed-B buffer
- aligned `weight_sums_oc` buffer

## Packed-B Layout

MMT4D RHS layout:

```text
n_tile = floor(output_channel / 4)
k_tile = floor(flat_k / 8)
offset = (n_tile * k_tiles + k_tile) * 32
B[n, k] stored as packed_b[offset + n * 8 + k]
```

Properties:

- output-channel tails are zero padded;
- K tails are zero padded;
- signed int8 weights are stored directly;
- selected real-node weights have zero-point `0`;
- weight sums are stored per output channel for correction.

Alignment:

- allocation alignment: `64` bytes

The format is runtime object state, not a committed model binary blob.
