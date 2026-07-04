# Split Branch Contract Report

## Split Node

node: `/model.2/Split`
input: `/model.2/cv1/act/Mul_output_0`
outputs:

- `/model.2/Split_output_0`: channels 0..15, deferred
- `/model.2/Split_output_1`: channels 16..31, selected

## Representation

The runner computes Conv2 activation into a full `[H,W,32]` NHWC buffer and copies channels `16..31` into a compact `[H,W,16]` NHWC buffer for `/model.2/m.0/cv1/conv/Conv`.

## Timing Bucket

The channel-slice copy is recorded as:

- `split_us`
- `pack_layout_us`

Stage 10 CPU0 full-shape result:

- `split_us`: `1130.39 us`
- `pack_layout_share`: `0.482372%`

## Decision

Split/copy is correct and not the current bottleneck. A view/strided Conv input path is deferred until a later stage because current Conv kernels expect compact NHWC.
