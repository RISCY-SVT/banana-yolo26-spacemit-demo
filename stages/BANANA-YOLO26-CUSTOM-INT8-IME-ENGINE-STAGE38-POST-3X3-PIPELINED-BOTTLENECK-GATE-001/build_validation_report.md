# Build Validation Report

## Host Native

- build dir: `.deps/custom_int8_engine/build-host-native-stage38`
- compiler: `/usr/bin/g++`
- IME: `OFF`
- build: `pass`
- CTest: `pass`, `42/42`
- CTest raw log: `/data/ncnn-logs/ai-team/2026-07-09_08-54-07/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001/run_logs/host_ctest_after_lane_a.log`

## RISC-V Cross

- build dir: `.deps/custom_int8_engine/build-riscv-stage38`
- compiler route: `/opt/riscv/bin/riscv64-unknown-linux-gnu-g++`
- flags: `-march=rv64gcv_zvfh -mabi=lp64d`
- IME: `ON`
- build: `pass`
- raw log: `/data/ncnn-logs/ai-team/2026-07-09_08-54-07/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001/run_logs/riscv_build_after_lane_a.log`

## Board

- board correctness: `pass`
- stable benchmark: `pass`
- FRM sweep: `pass`
- CPU affinity: `CPU0-3`
- CPU4-7 IME execution: `none`
