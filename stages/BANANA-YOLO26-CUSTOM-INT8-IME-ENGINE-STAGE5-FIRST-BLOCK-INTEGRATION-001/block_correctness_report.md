# Block Correctness Report

Selected block: `block0_conv_only`

Implementation files:

- `custom_int8_engine/include/y26_k1x_block_runner.h`
- `custom_int8_engine/src/block_runner.cpp`
- `custom_int8_engine/tests/test_stage5_first_block_runner.cpp`
- `custom_int8_engine/tests/stage5_block0_fixture.h`

## Host

Build:

- build dir: `.deps/custom_int8_engine/build-host-native-stage5`
- compiler: `/usr/bin/g++`
- `Y26_K1X_ENABLE_IME=OFF`

Result:

- CTest: `17/17` pass
- direct Stage 5 test:
  - `synthetic_seeded`: scalar status `0`, scalar mismatches `0`, IME status `1` because host build does not include IME
  - `synthetic_gradient`: scalar status `0`, scalar mismatches `0`, IME status `1` because host build does not include IME

## Board

Board:

- target: `svt@banana`
- kernel: `Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64`
- deployed dir: `/home/svt/yolo26-custom-int8-stage5/2026-07-03_22-07-15`

Smoke:

- runtime cached probe: pass
- Stage 1 `smt.vmadot` fixture: pass, total mismatches `0`
- Stage 4 packing repair fixture: pass

Stage 5 block correctness was run on CPU0, CPU1, CPU2, and CPU3 separately with `taskset`.

| CPU | case | scalar status | scalar mismatches | IME status | IME mismatches |
|---:|---|---:|---:|---:|---:|
| 0 | `synthetic_seeded` | `0` | `0` | `0` | `0` |
| 0 | `synthetic_gradient` | `0` | `0` | `0` | `0` |
| 1 | `synthetic_seeded` | `0` | `0` | `0` | `0` |
| 1 | `synthetic_gradient` | `0` | `0` | `0` | `0` |
| 2 | `synthetic_seeded` | `0` | `0` | `0` | `0` |
| 2 | `synthetic_gradient` | `0` | `0` | `0` | `0` |
| 3 | `synthetic_seeded` | `0` | `0` | `0` | `0` |
| 3 | `synthetic_gradient` | `0` | `0` | `0` | `0` |

No CPU4-7 IME execution was run in Stage 5.
