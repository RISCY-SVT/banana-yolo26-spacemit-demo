# Threading Correctness Report

Board correctness was run on the deployed RISC-V binaries.

## CPU0-3 Smoke

Commands ran:

```text
taskset -c <cpu> ./test_stage10_rvv_rounding_control
taskset -c <cpu> ./test_stage16_model4_c2f_runner
```

for CPU0, CPU1, CPU2, and CPU3.

Result:

```text
RNE/FRM regression: pass on CPU0/1/2/3
Stage16 model4 C2f smoke: pass on CPU0/1/2/3
```

## Stage18 Threaded Conv

Command:

```text
taskset -c 0-3 ./test_stage18_threaded_conv_integration
```

Result:

| thread_count | status | mismatches | checksum | worker_affinity_ok |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1324192976 | 1 |
| 2 | 0 | 0 | 1324192976 | 1 |
| 3 | 0 | 0 | 1324192976 | 1 |
| 4 | 0 | 0 | 1324192976 | 1 |

No CPU4-7 IME path was executed.
