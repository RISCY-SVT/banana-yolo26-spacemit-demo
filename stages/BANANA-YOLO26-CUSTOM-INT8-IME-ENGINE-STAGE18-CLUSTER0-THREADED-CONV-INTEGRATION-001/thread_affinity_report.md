# Thread Affinity Report

Board command:

```text
taskset -c 0-3 ./bench_stage18_threaded_conv_integration 10 100 5
```

Worker policy:

```text
worker 0 -> CPU0
worker 1 -> CPU1
worker 2 -> CPU2
worker 3 -> CPU3
```

The Stage18 API records worker entry CPU with `sched_getcpu()` after `pthread_setaffinity_np()` and before the hot-loop barrier.

Results:

```text
1 thread affinity: pass
2 thread affinity: pass
3 thread affinity: pass
4 thread affinity: pass
CPU4-7 IME execution: no
```

The main benchmark process was externally constrained to CPU0-3 and pinned its main thread to CPU0.
