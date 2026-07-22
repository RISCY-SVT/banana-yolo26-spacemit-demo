# Scheduler Lifecycle Repair

## Scope

This is a concurrency-maintenance repair. It does not change Conv, MatMul,
requantization, activation, layout, package, or model arithmetic.

## Invalid First Soak

The first R512/R384 soak started at `2026-07-20T13:16:50Z` and was terminated
at `2026-07-20T14:01:24Z`. It is invalid performance evidence and is not used
in `resolution_long_soak.tsv`.

At the stall, GDB showed the controller waiting in `WorkerPool::dispatch()`.
Three required workers were spinning, while one required worker remained
asleep in the frame-gated active-window condition variable. The process used
about 435% CPU but could not complete the four-worker dispatch. The O2 wrapper
restored its cgroup, IRQ, workqueue, and service state after SIGTERM.

## Root Cause And Repair

`begin_active_window()` published the active state and notified parked workers,
but did not wait for every worker to acknowledge the transition. A worker could
also observe an inactive window on the atomic fast path and park after the
controller had already issued the wake notification.

The repaired lifecycle:

1. serializes active-window transitions with the worker-pool mutex;
2. rechecks the active state under that mutex before a worker parks;
3. counts parked workers and waits for all of them to acknowledge park/wake;
4. rejects unchanged generations so a worker cannot replay the preceding job.

Threaded-convolution startup received a separate condition-variable repair:
worker readiness is now published while holding the readiness mutex, preventing
a creator-side lost wakeup between predicate evaluation and wait.

## Validation

- Host CTest after the active-window repair: 50/50 passed.
- ASan/UBSan after the repair: 51/51 passed.
- TSan suite: 50 tests passed within the 120-second per-test limit; the sole
  longer Stage48 test passed separately in 129.81 seconds with a 600-second
  limit and emitted no race report.
- Repeated focused TSan startup runs: 20/20 passed.
- RISC-V executor and camera-demo cross-builds passed.
- Board R512 lifecycle stress: 2,000/2,000 samples, normal exit, exact output
  `0x41e92be60b116000`, exact package manifest, and O2 restore passed.

The 2,000-run stress is lifecycle evidence, not the selected long-soak row.
Both official post-repair surfaces then completed 10,000/10,000 inferences:

- R512 mean 101.435319 ms, p95 101.327711 ms, p99 130.922900 ms,
  p99.9 132.127313 ms, maximum 132.743850 ms;
- R384 mean 48.014866 ms, p95 47.932396 ms, p99 59.929903 ms,
  p99.9 78.462473 ms, maximum 78.733062 ms.

The periodic between-block active-window wake cost is visible in the long-tail
columns and is retained rather than mixed with the separate 500-sample ABBA
surface. Both soaks held 1.6 GHz, preserved exact output, recorded zero affinity
failures and zero CPU4-7 IME operations, and restored O2 cleanly.
