# Block Correctness Report

## Host

Host-native build/CTest:

- build dir: `.deps/custom_int8_engine/build-host-native-stage10`
- tests: `27/27 passed`
- Stage 10 host mode: A0 int8 LUT scalar path, RVV-specific checks skipped on x86 host

## Board

Board target: `svt@banana`
Binaries deployed under a Stage 10 local board directory.
CPU affinity: CPU0, CPU1, CPU2, CPU3 separately.

### Rounding Control

For `test_stage10_rvv_rounding_control` on CPU0-3:

- synthetic_seeded mismatches: `0`
- synthetic_gradient mismatches: `0`
- ambient `frm` values tested: RNE, RTZ, RDN, RUP, RMM

### Expanded Subset

For `test_stage10_backbone_expansion_runner` on CPU0-3:

- scalar A0: `act2_mismatches=0`, `split_mismatches=0`, `branch0_mismatches=0`
- scalar A2 on RISC-V: `act2_mismatches=0`, `split_mismatches=0`, `branch0_mismatches=0`
- IME A2: `act2_mismatches=0`, `split_mismatches=0`, `branch0_mismatches=0`

## Decision

`block_oracle: pass`, `host_tests: pass`, `board_tests: pass` for the selected Stage 10 expanded subset.
