# Block Correctness Report

classification: `pass`

## Host

Host-native CTest: `19/19` pass with `Y26_K1X_ENABLE_IME=OFF`.

## Board

Board: `Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64`.
Online CPUs: `0-7`.
IME tests were pinned only to CPU0-3.

| CPU | scalar status | IME status | scalar mismatches | IME mismatches |
|---:|---:|---:|---:|---:|
| 0 | `0` | `0` | `0` | `0` |
| 1 | `0` | `0` | `0` | `0` |
| 2 | `0` | `0` | `0` | `0` |
| 3 | `0` | `0` | `0` | `0` |

Each CPU ran both deterministic fixtures: `synthetic_seeded` and `synthetic_gradient`. Checked boundaries: Conv0 int32, Act0 signed handoff, Conv1 int32, Act1 signed handoff, Conv2 int32.

Raw log: `run_logs/025_board_stage7_correctness_and_bench_rerun.txt`.
