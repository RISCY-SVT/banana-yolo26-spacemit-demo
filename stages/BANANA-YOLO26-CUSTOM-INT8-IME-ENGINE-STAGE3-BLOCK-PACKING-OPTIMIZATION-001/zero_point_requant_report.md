# Zero-Point And Requant Boundary Report

## Implemented Scope

Stage 3 implements real selected-node correction for `uint8` activation tensors stored as signed bytes:

```text
a_s8 = q_u8 - 128
raw = sum(a_s8 * w_s8)
corrected = raw + (128 - za) * sum(w_s8) + bias_i32
```

This is equivalent to:

```text
sum((q_u8 - za) * (w_s8 - zw)) + bias_i32
```

for selected nodes where `zw = 0`.

## Selected Nodes

| node | activation zp | weight zp | correction status |
|---|---:|---:|---|
| `/model.2/cv1/conv/Conv` | `0` | all `0` | implemented |
| `/model.2/m.0/cv1/conv/Conv` | `2` | all `0` | implemented |

Padding for quantized Conv3x3 uses storage value `za - 128`, not raw signed zero.

## Requant Boundary

- Stage 3 compares corrected `int32` accumulators through dequantized float equivalence against ORT CPU Conv outputs.
- No full graph requant policy is claimed.
- No activation function, scheduler, or downstream graph integration is implemented.

## Caveat

Asymmetric weight zero-points are not needed for the selected nodes. If a later node has nonzero `zw`, the full correction term must be added:

```text
sum(q*w) - zw*sum(q) - za*sum(w) + K*za*zw
```
