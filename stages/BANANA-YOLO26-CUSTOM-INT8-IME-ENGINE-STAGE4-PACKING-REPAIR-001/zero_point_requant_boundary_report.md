# Zero-Point and Requant Boundary Report

Stage 4 did not move zero-point correction into the `smt.vmadot` microkernel.

Raw kernels compute:

```text
acc = sum(s8_activation_storage * s8_weight)
```

For selected real nodes, activations are stored as:

```text
s8_activation_storage = q_u8 - 128
```

Selected real weights are signed int8 with zero-point `0`.

Correction used by `y26_conv2d_apply_u8_as_s8_correction_nhwc`:

```text
corrected = raw + (128 - activation_zero_point_u8) * sum(weight_s8) + bias_i32
```

This is the selected-node specialization of:

```text
sum((a - za) * (w - zw))
= sum(a*w) - zw*sum(a) - za*sum(w) + K*za*zw
```

Requant/dequant policy remains a graph-integration boundary. Stage 4 compares selected real-node integer fixtures and does not claim full graph requant correctness.
