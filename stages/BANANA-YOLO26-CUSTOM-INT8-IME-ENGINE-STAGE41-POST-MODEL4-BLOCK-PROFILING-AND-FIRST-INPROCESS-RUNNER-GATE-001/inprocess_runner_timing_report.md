# In-Process Runner Timing Report

Timing is scaffold profiling only. It is not model FPS and not production latency.

## Host Exact Scaffold

```text
custom_mode: scalar
warmup: 1
runs: 3
repeats: 2
mean_total_us: 301182.315667
stddev_total_us: 5487.214147
cv_total_pct: 1.821891
mean_prefix_us: 60853.329333
mean_custom_model4_us: 96848.580667
mean_suffix_us: 130142.634167
mean_layout_conversion_us: 1631.365500
mean_attribution_pct: 96.113183
```

## Board Selected-Mode Attempt

```text
custom_mode: ime_threaded
warmup: 10
runs: 100
repeats: 5
mean_total_us: 858404.224484
stddev_total_us: 1433.217623
cv_total_pct: 0.166963
mean_prefix_us: 230282.265274
mean_custom_model4_us: 25354.844728
mean_suffix_us: 554279.618664
mean_layout_conversion_us: 11022.462626
mean_attribution_pct: 95.635502
correctness: fail
```

The board timing is recorded as blocked evidence only because output0 was not byte-exact.
