# Expanded A-Panel Layout Report

Primary node:

`/model.4/m.0/cv1/conv/Conv`

Shape:

`80x80x32 -> 80x80x16`, kernel `3x3`, stride `1`, pad `1`

Direct sidecar layout:

- Builds an 8-row by padded-K A-panel for each output panel.
- `K = 3 * 3 * 32 = 288`.
- `K` is already a multiple of 8.
- Panel storage is M-major inside each 8-lane K tile to match the existing `smt.vmadot` 4x4x8 helper contract.
- Padding rows/columns use `input_storage_zero_point_s8`.

Schedule:

- Base `smt.vmadot` computes rows 0..3.
- `smt.vmadot1` computes shifted rows; only row 3 is kept as output row 4.
- `smt.vmadot2` computes shifted rows; only row 3 is kept as output row 5.
- `smt.vmadot3` computes shifted rows; only row 3 is kept as output row 6.

Reason for this conservative layout:

The layout directly exercises the Stage30-proven shifted-A-row semantics on a real Conv node without changing the existing accepted MMT4D runner path.

Measured issue:

Panel construction dominates the direct candidate:

`panel_build_mean_us=38901.3` out of `direct_mean_us=56980.9`.
