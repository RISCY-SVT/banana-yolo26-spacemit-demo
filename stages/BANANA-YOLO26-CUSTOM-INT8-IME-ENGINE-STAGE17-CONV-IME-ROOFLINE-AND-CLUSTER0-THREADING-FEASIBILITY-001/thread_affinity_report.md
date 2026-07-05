# Thread Affinity Report

Board execution policy:

```text
Allowed IME CPUs: CPU0, CPU1, CPU2, CPU3
Forbidden IME CPUs: CPU4, CPU5, CPU6, CPU7
```

Stage17 benchmark invocation:

```text
taskset -c 0-3 ./bench_stage17_roofline_threading 10 100 5
```

The benchmark pins:

```text
main thread: CPU0
1-thread worker: CPU0
2-thread workers: CPU0, CPU1
3-thread workers: CPU0, CPU1, CPU2
4-thread workers: CPU0, CPU1, CPU2, CPU3
```

No Stage17 IME path was run on CPU4-7.
