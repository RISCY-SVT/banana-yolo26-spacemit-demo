# Stage 14 Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001`
replayed_stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001`
candidate: `candidate_H3_model2_act_model3_act_model4_cv1_conv`
start_head: `5cc09059f83eaef6af8c9a6aee3eab1e4edd46e7`

## Host Baseline

`ctest --test-dir .deps/custom_int8_engine/build-host-native-stage15 --output-on-failure`

Result: `31/31` tests passed.

## RISC-V Cross Baseline

Build route:

`source /data/build_scripts/01-env.sh; cmake -S custom_int8_engine -B .deps/custom_int8_engine/build-riscv-stage15 -DCMAKE_BUILD_TYPE=Release -DY26_K1X_ENABLE_IME=ON -DCMAKE_CXX_COMPILER=${CXX:-riscv64-unknown-linux-gnu-g++} -DCMAKE_CXX_FLAGS="${K1_ARCH_FLAGS:--march=rv64gcv_zvfh -mabi=lp64d}" -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64`

Result: cross build passed.

RISC-V binary SHA256:

| binary | sha256 |
|---|---|
| `tests/test_stage10_rvv_rounding_control` | `6d8403c56bf1ad542b65f45a461a7865e12faf7574954f21edd83c8717be2f3d` |
| `tests/test_stage14_next_c2f_runner` | `7068be2a131fecd882f1c26e1904294ec0b3b7d25d87a92d7989d9e3d2473e86` |
| `bench_stage14_next_c2f` | `066979588559a3f09f0f30e5a16eef0fbdbcf147e06b60d48b9ce079b0f9c80e` |

## Board Replay

Board target: `svt@banana`
board dir: `/home/svt/yolo26-custom-int8-stage15/2026-07-05_13-30-42/baseline`

CPU0/1/2/3 correctness replay passed for:

- `test_stage10_rvv_rounding_control`
- `test_stage14_next_c2f_runner`

All reported mismatch counters were `0`.

## CPU0 Compact Microbench Replay

Command:

`taskset -c 0 ./bench_stage14_next_c2f 3`

| candidate | total_us | conv_us | activation_requant_us | merge_us | pack_layout_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | `345.396` | `244.63` | `77.1413` | `13.0833` | `0.389` | `70.8258` | `22.3341` | `3.78792` | `0.112624` | `0` |
| `stage14_IME_A2_rvv_f32_lut` | `140.518` | `97.293` | `22.195` | `12.0973` | `0.347333` | `69.239` | `15.7952` | `8.60912` | `0.247181` | `0` |

## Status

stage14_baseline_rechecked: `pass`
