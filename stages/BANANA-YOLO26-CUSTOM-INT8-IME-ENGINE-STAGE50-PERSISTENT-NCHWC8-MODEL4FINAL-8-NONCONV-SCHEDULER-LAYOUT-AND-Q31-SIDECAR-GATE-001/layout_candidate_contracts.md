# Layout candidates

L0 is `NCHWc8_SPATIAL_INNER_V1`. Bounded sidecar contracts also define L1 `M4C8_KMAJOR_V1`, L2 `NCHWc16_SPATIAL_INNER_V1`, and L3 NHWC16, each with deterministic offsets, alignment, and tails. Only L0 currently has persistent producer and consumer kernels across the tested Conv pairs.
