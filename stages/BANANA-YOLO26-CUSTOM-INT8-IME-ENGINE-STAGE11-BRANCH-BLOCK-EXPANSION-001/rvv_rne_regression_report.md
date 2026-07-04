# RVV RNE Regression Report

selected_mode: `A2_rvv_f32_lut`

## Host

Host-native CTest passed. RVV runtime tests are skipped on the host because host-native build is `Y26_K1X_ENABLE_IME=OFF` and no RISC-V vector execution is available.

## Board CPU0-3

`test_stage10_rvv_rounding_control` was run on CPU0, CPU1, CPU2, and CPU3. For both compact fixtures it tested ambient `frm` values `0..4`.

Result:

- status: pass
- mismatches: `0`
- after-call `frm`: preserved for each ambient mode

Stage 11 did not change the accepted explicit RNE path in `activation_requant.cpp`.
