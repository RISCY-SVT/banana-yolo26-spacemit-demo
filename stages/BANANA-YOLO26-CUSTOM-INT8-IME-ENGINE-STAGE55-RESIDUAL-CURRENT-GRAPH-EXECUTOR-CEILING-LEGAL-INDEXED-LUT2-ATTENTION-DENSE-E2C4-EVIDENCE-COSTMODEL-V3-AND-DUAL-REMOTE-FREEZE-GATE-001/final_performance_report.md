# Final performance

Compatibility SCHED_OTHER mean 162447.546000 us, p95 166485.750000 us. Frame-gated low-latency SCHED_OTHER mean 149603.240000 us, p95 153748.300000 us, p99 159241.830000 us.

The selected route is 10.637597% lower latency than official Stage54 and 67.122202% lower than matched B120 ORT. The 100-image real-corpus mean is 149183.650000 us; preloaded-image pipeline mean is 172265.633994 us.

The separate 10000-run soak records mean 149598.790900 us, p95 152636.150000 us, p99 158975.130000 us, p99.9 203354.276000 us, max 211671.000000 us. These statistics are not mixed into the 500-run headline row.
