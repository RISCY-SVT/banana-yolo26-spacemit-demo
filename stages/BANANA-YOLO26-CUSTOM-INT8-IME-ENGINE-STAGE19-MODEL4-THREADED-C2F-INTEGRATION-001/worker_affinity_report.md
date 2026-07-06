# Worker Affinity Report

Policy:

```text
IME worker CPUs allowed: CPU0, CPU1, CPU2, CPU3
IME worker CPUs forbidden: CPU4, CPU5, CPU6, CPU7
threading backend: explicit cluster0 worker pool
OpenMP/default all-core dispatch: not used
```

Observed:

```text
Stage18 representative replay worker_affinity_ok: 1 for A0/A1/A2/A3/A4
Stage19 compact correctness worker affinity: ok for thread counts 1/2/3/4 and activation sidecar
Stage19 stable compact bench affinity_ok: 1 for all candidates
CPU4-7 IME execution: none
```

The implementation keeps threaded modes explicit. The default scalar and existing IME hotpaths remain unchanged.
