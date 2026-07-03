# Stage 3 Baseline Recheck

commit: `2d0fd778619aa14189921905d1ba36afc11102ff`
branch: `yolo26-custom-int8-engine`

## Host

Command set:

- `cmake -S custom_int8_engine -B .deps/custom_int8_engine/build-host-native-stage4-baseline -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++`
- `cmake --build .deps/custom_int8_engine/build-host-native-stage4-baseline -j$(nproc)`
- `ctest --test-dir .deps/custom_int8_engine/build-host-native-stage4-baseline --output-on-failure`

Result: `15/15` pass.

## Cross Build

Toolchain route: `/data/build_scripts/01-env.sh`

- compiler: `/opt/riscv/bin/riscv64-unknown-linux-gnu-g++`
- flags: `-march=rv64gcv_zvfh -mabi=lp64d`
- `Y26_K1X_ENABLE_IME=ON`

Result: pass.

## Board

Target: `svt@banana`
Kernel: `Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64`
Affinity: `taskset -c 0-3` for smoke, `taskset -c 0` for baseline bench.

Correctness smoke:

- runtime probe: pass
- `smt.vmadot` fixture: pass
- Conv1x1 synthetic fixture: pass
- Conv3x3 synthetic fixture: pass
- Stage 3 real Conv fixture: pass

Baseline microbench, `bench_stage3_packing 3`:

| case | scalar us | old wrapper us | Stage 3 prepacked us | packA/im2col probe us | correction us |
|---|---:|---:|---:|---:|---:|
| Conv1x1 `160x160x32->32` | `109547` | `133422` | `46649.6` | `33111.1` | `2656.46` |
| Conv3x3 `160x160x16->8` | `143708` | `389408` | `149121` | `157570` | `661.189` |

Baseline reproduced: yes.
