# Cluster0 Threading Feasibility Note

Stage 15 does not enable multithreading by default.

Policy:

- IME kernels are cluster0 CPU0-3 only.
- No CPU4-7 IME execution.
- No OpenMP/all-core default dispatch.
- Do not benchmark compact fixture threading as headline evidence.
- Controlled threading feasibility should wait until representative/full-shape block/subgraph is stable.

Future candidate tests:

| test | affinity | condition |
|---|---|---|
| `1_thread_cpu0` | `CPU0` | representative/full-shape block stable |
| `2_threads_cpu0_1` | `CPU0-1` | no hot-loop allocation and explicit work split |
| `3_threads_cpu0_2` | `CPU0-2` | same |
| `4_threads_cpu0_3` | `CPU0-3` | same |
| `cpu4_negative_guard` | `CPU4` | only if safe and already authorized by a bounded guard |

threading_default_authorized: `false`
