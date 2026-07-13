# Full-graph LUT-v2 estimate

The exact accepted graph contains 106 compute rows and 2,740,153,600 MACs. Stage51 maps
96.172127% of MACs to stable full-shape integrated rows, including the historical
exact full-shape RGB stem, and maps all material non-MAC classes to measured or explicitly
conservative rows. Unsupported small-N, grouped/depthwise, and attention MatMul rows remain
visible rather than inheriting model5 throughput.

- optimistic: 158.973694 ms
- central: 204.380817 ms
- conservative: 269.869364 ms

These are analytical execution envelopes, not measured full-model latency or hardware bounds.
Even the optimistic envelope is far above the 45 ms pure-model target. The maximized current
graph is therefore `current-graph-not-target-credible-on-maximized-substrate`.
