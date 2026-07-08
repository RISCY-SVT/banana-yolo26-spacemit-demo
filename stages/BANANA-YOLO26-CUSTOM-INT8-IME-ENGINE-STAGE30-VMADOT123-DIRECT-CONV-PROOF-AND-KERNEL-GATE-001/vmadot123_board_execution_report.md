# vmadot123 Board Execution Report

Board policy:

- IME execution limited to CPU0, CPU1, CPU2, CPU3.
- CPU4-7 were not used for IME execution.
- No negative CPU4-7 execution was performed in this stage.

Current board execution proof used `bench_stage30_vmadot123_probe --derive-oracle` under `taskset -c <cpu>`.

| CPU | status_failures | traps | validation_mismatches | result |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | pass |
| 1 | 0 | 0 | 0 | pass |
| 2 | 0 | 0 | 0 | pass |
| 3 | 0 | 0 | 0 | pass |

The initial 32-byte A-window probe also executed without traps, but failed oracle validation because `vmadot1/2/3` read shifted A rows outside the 4x8 base tile. Stage30 corrected this by using a proof-only 8x8 A window and non-overlapping vector register groups.
