# Board Test Report

Board:

```text
Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
CPU(s): 8
Model name: Spacemit(R) X60
```

Remote directory:

```text
/home/svt/yolo26-custom-int8-stage2/2026-07-03_12-27-20
```

No Stage 2 test intentionally ran IME on CPU4-7.

## Probe

| CPU | command | status |
| ---: | --- | --- |
| 0 | `taskset -c 0 ./test_ime_runtime_probe` | pass |
| 1 | `taskset -c 1 ./test_ime_runtime_probe` | pass |
| 2 | `taskset -c 2 ./test_ime_runtime_probe` | pass |
| 3 | `taskset -c 3 ./test_ime_runtime_probe` | pass |

## Correctness Fixtures

| case | command | status | mismatches |
| --- | --- | --- | ---: |
| direct `smt.vmadot` CPU0 | `taskset -c 0 ./test_vmadot_4x4x8_board_probe` | pass | 0 |
| direct `smt.vmadot` CPU1 | `taskset -c 1 ./test_vmadot_4x4x8_board_probe` | pass | 0 |
| direct `smt.vmadot` CPU2 | `taskset -c 2 ./test_vmadot_4x4x8_board_probe` | pass | 0 |
| direct `smt.vmadot` CPU3 | `taskset -c 3 ./test_vmadot_4x4x8_board_probe` | pass | 0 |
| Conv1x1 fixture | `taskset -c 0-3 ./test_conv1x1_ime_fixture` | pass | 0 |
| Conv3x3 fixture | `taskset -c 0-3 ./test_conv3x3_ime_fixture` | pass | 0 |

## Benchmarks

Benchmarks are kernel-level only and are not YOLO26 inference benchmarks.

| case | command | status |
| --- | --- | --- |
| hotpath vmadot | `taskset -c 0 ./bench_vmadot_hotpath 20000 5` | pass |
| Conv kernels | `taskset -c 0-3 ./bench_conv_kernels 200 5` | pass |
