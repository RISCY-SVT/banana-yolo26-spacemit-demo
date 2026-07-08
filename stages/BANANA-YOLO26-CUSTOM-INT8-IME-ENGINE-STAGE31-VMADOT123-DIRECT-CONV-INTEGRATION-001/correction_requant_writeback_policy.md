# Correction/Requant/Writeback Policy

Stage31 direct sidecar output boundary:

- Corrected int32 output for `/model.4/m.0/cv1/conv/Conv`.
- It does not perform downstream activation/QDQ.
- It compares against the existing accepted full-shape branch-entry reference for the same node.

Correction:

- Uses the existing signed-storage zero-point correction helper:
  `y26_conv2d_apply_u8_as_s8_correction_nhwc`
- Bias and weight sums are taken from existing prepacked Conv weights.

Writeback:

- Raw direct accumulators are first written into a workspace int32 buffer.
- Corrected output is written to the caller-provided NHWC int32 output.

Timing:

| Bucket | mean_us |
| --- | ---: |
| panel_build | 38901.3 |
| kernel_compute | 15795.9 |
| correction | 201.322 |
| writeback | 1275.35 |
| direct_total | 56980.9 |

Conclusion:

Correction is not the blocker. Panel build and duplicate-row direct compute dominate.
