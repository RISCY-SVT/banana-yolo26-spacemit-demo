# Layout Adapter Decision

The correctness-oracle scalar adapters were benchmarked separately with preallocated buffers.

- island entry, NCHW uint8 `1x64x80x80` to NHWC uint8: `3079.934 us`, 0.266 GB/s effective read+write throughput;
- island exit, NHWC signed-code `1x128x40x40` to NCHW uint8: `801.232 us`, 0.511 GB/s.

There is no adapter between model4 and model5. No RVV/tiled adapter was implemented because model5 failed the earlier internal-compute gate; adapter optimization cannot make this model5 candidate competitive by itself.
