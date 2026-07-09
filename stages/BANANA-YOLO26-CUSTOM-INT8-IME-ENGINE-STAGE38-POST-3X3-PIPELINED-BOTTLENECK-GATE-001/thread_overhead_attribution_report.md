# Thread Overhead Attribution Report

## Policy

`thread_overhead_us` is a diagnostic sub-bucket from the cluster0 threaded Conv path. It is already included inside `conv_us`, so it is not added again in selected-cut top-level attribution.

## Measurements

| mode | thread_overhead_us | conv_us | thread_overhead_share_of_conv_pct | selected_total_us |
|---|---:|---:|---:|---:|
| Stage37 replay, `rvv` output quantize | 4707.19 | 18051.0 | 26.08 | 32890.5 |
| Stage38 Lane A, `rvv_direct` output quantize | 4730.13 | 18048.1 | 26.21 | 30341.5 |

## Decision

Thread overhead is material, but Stage38 decision tree selected Lane A first because output QuantizeLinear exceeded the explicit lane threshold and had a bounded exact repair. A future persistent-thread/region stage remains possible, but the next immediate bucket after Lane A is branch 3x3 im2col/pack.
