# Workspace Reuse Report

Object: `Y26ConvWorkspace`

The public type is opaque. The implementation stores:

- original `Y26Conv2DParams`
- `kernel_h`
- `kernel_w`
- `k_padded`
- `bytes`
- `peak_bytes`
- aligned A-tile buffer

## A Tile Layout

MMT4D LHS tile layout:

```text
k_tile = floor(flat_k / 8)
k_lane = flat_k % 8
offset = k_tile * 32 + m * 8 + k_lane
A[m, flat_k] stored as workspace[offset]
```

This lets the hot loop pass `workspace + k_tile * 32` directly to the proven `smt.vmadot 4x4x8` microkernel.

## Selected Real Shapes

| case | workspace bytes | peak bytes |
|---|---:|---:|
| Conv1x1 `160x160x32->32` | `128` | `128` |
| Conv3x3 `160x160x16->8` | `576` | `576` |

No heap allocation is performed inside the repeated Conv hot loop.
