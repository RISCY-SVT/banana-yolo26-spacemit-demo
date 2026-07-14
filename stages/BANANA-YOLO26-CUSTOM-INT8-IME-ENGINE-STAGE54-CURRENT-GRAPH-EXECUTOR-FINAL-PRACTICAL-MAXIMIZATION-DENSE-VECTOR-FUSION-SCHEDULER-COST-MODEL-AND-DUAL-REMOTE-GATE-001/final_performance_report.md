# Final performance

Compatibility condition-variable SCHED_OTHER: 180237.700000 us mean, 182678.350000 us p95. Dedicated-board epoch-spin SCHED_OTHER: 167411.836000 us mean, 169621.050000 us p95, 173464.770000 us p99.

The low-latency route is 30.211342% lower latency than Stage53 and 63.444709% lower than matched B120 ORT on the same per-inference statistical unit.

Real 100-image corpus mean is 166140.750000 us. Preloaded-image pipeline mean is 190754.405260 us.

The 10000-run soak recorded p99 177433.070000 us, p99.9 182884.190000 us, and max 186348.000000 us.

The separate 10000-run compatibility soak recorded mean 180403.108600 us, p99 190387.200000 us, and max 214786.000000 us.
