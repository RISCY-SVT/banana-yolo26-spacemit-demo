# Correctness Report

classification: pass

## Host

- Build dir: `.deps/custom_int8_engine/build-host-native-stage9`
- CMake: host `/usr/bin/g++`, `Y26_K1X_ENABLE_IME=OFF`
- CTest: `25/25` passed.

New Stage 9 tests:

- `test_stage9_requant_scalar_optimized`
- `test_stage9_activation_modes`
- `test_stage9_lut_oracle`
- `test_stage9_pack_handoff`

## Cross

- Build dir: `.deps/custom_int8_engine/build-riscv-stage9`
- `Y26_K1X_ENABLE_IME=ON`
- Result: pass.

## Board

Board: `svt@banana`
Cluster policy: CPU0-3 only.

| test | CPU0 | CPU1 | CPU2 | CPU3 |
|---|---|---|---|---|
| `test_stage9_requant_scalar_optimized` | pass | pass | pass | pass |
| `test_stage9_activation_modes` | pass | pass | pass | pass |
| `test_stage9_lut_oracle` | pass | pass | pass | pass |
| `test_stage9_pack_handoff` | pass | pass | pass | pass |

Accepted exact paths with `mismatches=0`:

- A1 scalar-unrolled LUT.
- A2 RVV f32 LUT.
- A3 fixed-requant LUT.
- A4 fused current-layout alias.

No CPU4-7 IME tests were run.
