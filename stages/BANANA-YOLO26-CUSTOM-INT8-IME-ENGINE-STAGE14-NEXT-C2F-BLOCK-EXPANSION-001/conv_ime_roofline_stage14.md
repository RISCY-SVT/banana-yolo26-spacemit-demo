# Stage 14 Conv/IME Roofline Diagnostic

This is selected-subset diagnostic evidence only. It is not full-model roofline and not a production claim.

## Newly Covered Conv Nodes

| node | compact shape | kernel | MACs in compact fixture | IME conv component us | rough GMAC/s | classification |
|---|---|---|---:|---:|---:|---|
| `/model.3/conv/Conv` | `[2,2,64] -> [1,1,64]` | `3x3 stride2 pad1` | `36864` | `38.0727` | `0.968` | structurally-small-K/fixture-bound |
| `/model.4/cv1/conv/Conv` | `[1,1,64] -> [1,1,64]` | `1x1 stride1` | `4096` | `7.09833` | `0.577` | structurally-small-K/fixture-bound |

## Decision

The compact Stage 14 fixture is too small for peak utilization conclusions. The diagnostic only says that Conv is now the dominant bucket in the selected compact H3 path. Future full-shape block work should keep Conv/IME roofline visible and may justify a separate direct-conv/sliding feasibility stage, but Stage 14 does not implement `vmadot1/2/3`.
