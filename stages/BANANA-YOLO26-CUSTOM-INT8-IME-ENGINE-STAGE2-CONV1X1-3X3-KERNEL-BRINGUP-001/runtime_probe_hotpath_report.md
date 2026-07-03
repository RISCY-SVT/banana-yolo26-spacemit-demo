# Runtime Probe And Hotpath Report

## Scope

Stage 2 replaced the Stage 1 per-call SIGILL guard path with:

- one-time process-level `y26_k1x_ime_probe_once()`;
- cached `Y26_IME_CAPABILITY_AVAILABLE_CLUSTER0_ONLY` state;
- dispatch-boundary CPU0-3 check via `y26_k1x_ime_hotpath_allowed_on_current_cpu()`;
- public cached wrapper `y26_vmadot_4x4x8_ime_s8s8s32()`;
- internal `y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32()` for already-pinned tile loops.

The unsafe entrypoint is not a model API. It assumes the caller has pinned or otherwise constrained execution to cluster0.

## Board Probe

Command scope: board, `taskset -c <cpu> ./test_ime_runtime_probe`.

| CPU | first_probe | second_probe | capability | probe_status |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 2 | 0 |
| 1 | 0 | 0 | 2 | 0 |
| 2 | 0 | 0 | 2 | 0 |
| 3 | 0 | 0 | 2 | 0 |

`capability=2` means `Y26_IME_CAPABILITY_AVAILABLE_CLUSTER0_ONLY`.

## Hotpath Benchmark

This is a microkernel-only benchmark, not a YOLO26 inference benchmark.

Command:

```bash
taskset -c 0 ./bench_vmadot_hotpath 20000 5
```

| path | packing included | mean ns/call | stddev ns/call | speedup vs scalar |
| --- | --- | ---: | ---: | ---: |
| scalar | no | 251.982 | 0.875 | 1.000 |
| public cached wrapper | no | 95.453 | 0.320 | 2.640 |
| unsafe cluster0 | no | 60.178 | 0.372 | 4.187 |

The public cached wrapper no longer installs SIGILL guards or calls `sched_getcpu()` on every microkernel invocation after the thread has passed the cluster0 boundary check.
