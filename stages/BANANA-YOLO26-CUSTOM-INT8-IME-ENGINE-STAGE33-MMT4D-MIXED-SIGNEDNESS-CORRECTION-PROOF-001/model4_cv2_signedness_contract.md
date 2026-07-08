# Model4 CV2 Signedness Contract

target_node: `/model.4/cv2/conv/Conv`
shape: `80x80x96 -> 80x80x128`
kernel: `1x1`
MAC_count: `78643200`

## Baseline

The accepted MMT4D path stores activation input as signed int8 storage:

```text
storage_s8 = q_u8 - 128
dot = sum(storage_s8 * weight_s8)
corrected = dot + (128 - activation_zero_point_u8) * sum(weight_s8) + bias
```

For this boundary:

```text
activation_zero_point_u8: 15
input_storage_zero_point_s8: -113
weight_dtype: int8
bias_dtype: int32
```

## Mixed Signedness Candidate

Candidate operand contract:

```text
A: activation q_u8, reconstructed while packing from signed storage
B: weight_s8
instruction: smt.vmadotus
dot = sum(q_u8 * weight_s8)
corrected = dot - activation_zero_point_u8 * sum(weight_s8) + bias
```

The analytic correction term is per-output-channel constant:

```text
adjusted_bias[oc] = bias[oc] - activation_zero_point_u8 * weight_sum[oc]
```

Stage33 applies this adjusted bias directly when initializing each 4x4 accumulator tile, producing corrected int32 output without the separate baseline correction pass.
