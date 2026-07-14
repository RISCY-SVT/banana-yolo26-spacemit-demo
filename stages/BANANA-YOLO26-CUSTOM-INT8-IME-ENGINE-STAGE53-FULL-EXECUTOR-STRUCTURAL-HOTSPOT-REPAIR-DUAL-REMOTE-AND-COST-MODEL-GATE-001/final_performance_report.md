# Final performance

The selected SCHED_OTHER epoch-spin research route measured 239884.016000 us mean, 242452.000000 us p95, and 250782.090000 us p99 over 500 per-inference samples.

This is 2.099500x faster than the reproduced Stage52 executor and 1.902029x faster than matched B120 ORT using the same per-inference unit.

The 10,000-run selected-route soak measured 240043.539700 us mean, 242403.200000 us p95, 250091.230000 us p99, 260405.010000 us p99.9, and 263494.000000 us maximum.

The separate preloaded-image pipeline measured 253694.519464 us mean over 500 samples.

The condition-variable route remains the compatibility default. The epoch-spin route is an explicitly selected optimized-research mode with higher process CPU occupancy.
