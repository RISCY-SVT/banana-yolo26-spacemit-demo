# Stage 8 Baseline Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE9-RVV-REQUANT-LUT-FUSION-GATE-001`
baseline_stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001`
start_head: `71e143271b2d09eb35511725e360c3c95bddfc09`
classification: pass

## Host Baseline

- Build dir: `.deps/custom_int8_engine/build-host-native-stage9-baseline`
- CMake: host `/usr/bin/g++`, `Y26_K1X_ENABLE_IME=OFF`
- CTest: `21/21` passed before Stage 9 code changes.

## Cross Baseline

- Build dir: `.deps/custom_int8_engine/build-riscv-stage9-baseline`
- Compiler: `/opt/riscv/bin/riscv64-unknown-linux-gnu-g++`
- GCC: `14.3.0`
- Flags: `-march=rv64gcv_zvfh -mabi=lp64d`
- Result: pass.

## Board Baseline Replay

Board: `svt@banana`
CPU affinity: `taskset -c 0`

| mode | total_us | activation_us | activation_share | mismatches | checksum |
|---|---:|---:|---:|---:|---:|
| `scalar_float_reference` | 1258200 | 465984 | 37.04% | 0 | 707794080 |
| `ime_scalar_float_reference` | 620581 | 465664 | 75.04% | 0 | 707794080 |
| `ime_int8_lut` | 350320 | 193436 | 55.2171% | 0 | 707794080 |

The replay is within the expected Stage 8 range and is suitable as the Stage 9 before baseline.

Evidence logs:

- `$LOG_DIR/run_logs/host_stage9_baseline_ctest.log`
- `$LOG_DIR/run_logs/cross_stage9_baseline_build.log`
- `$LOG_DIR/run_logs/board_stage9_baseline_cpu0_bench.log`
