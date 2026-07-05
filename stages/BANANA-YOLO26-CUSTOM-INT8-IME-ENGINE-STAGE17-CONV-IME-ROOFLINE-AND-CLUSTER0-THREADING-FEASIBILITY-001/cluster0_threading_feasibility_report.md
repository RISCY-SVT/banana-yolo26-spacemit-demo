# Cluster0 Threading Feasibility Report

Protocol: `warmup=10 runs=100 repeats=5`

Gate:

```text
4-thread mean speedup >= 1.5x: accepted_for_future_integration
>= 2.0x: positive
>= 3.0x: strong_positive
```

Result:

```text
threading_feasibility: strong_positive
4_thread_speedup_vs_1thread: 3.680290
4_thread_mean_us: 5583.807628
1_thread_mean_us: 20550.030600
mismatches: 0
```

Decision: Stage17 supports opening Stage18 for bounded cluster0 threaded Conv integration. This must remain CPU0-3 only and must not become a default all-core/OpenMP dispatch.
