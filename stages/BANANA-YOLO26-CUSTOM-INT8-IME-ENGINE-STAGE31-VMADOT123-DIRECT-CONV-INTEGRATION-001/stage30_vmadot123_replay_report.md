# Stage30 vmadot1/2/3 Replay Report

Replayed binary:

`/home/svt/banana-yolo26-stage31-vmadot123-direct/bench_stage30_vmadot123_probe`

Binary SHA256:

`e2069af7208af3fd9b2fdf21f5692fae7a3822bee294c9ea168e6feeaa7e50ca`

Command pattern:

`taskset -c <cpu> bench_stage30_vmadot123_probe --derive-oracle`

CPU coverage:

| CPU | exit | total_status_failures | total_traps | total_validation_mismatches |
| --- | ---: | ---: | ---: | ---: |
| CPU0 | 0 | 0 | 0 | 0 |
| CPU1 | 0 | 0 | 0 | 0 |
| CPU2 | 0 | 0 | 0 | 0 |
| CPU3 | 0 | 0 | 0 | 0 |

Result:

`smt.vmadot1`, `smt.vmadot2`, and `smt.vmadot3` remain board-executable and oracle-clean on cluster0 CPU0-3.

Raw logs:

- `stage30_replay_cpu0.log`
- `stage30_replay_cpu1.log`
- `stage30_replay_cpu2.log`
- `stage30_replay_cpu3.log`
