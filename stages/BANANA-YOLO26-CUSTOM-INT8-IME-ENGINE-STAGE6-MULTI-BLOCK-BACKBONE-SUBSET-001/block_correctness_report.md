# Block Correctness Report

Selected subset: `candidate_C_block0_silu_model1_conv`

Boundary:

```text
/model.0/conv/Conv
Conv0 Q/DQ
/model.0/act/Sigmoid
/model.0/act/Mul
Act0 Q/DQ
/model.1/conv/Conv
```

Output boundary: corrected int32 output of `/model.1/conv/Conv`.

## Host

Host build:

```text
cmake -S custom_int8_engine -B .deps/custom_int8_engine/build-host-native-stage6 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DY26_K1X_ENABLE_IME=OFF
```

CTest: `18/18` pass.

Direct Stage 6 test:

| case | scalar status | scalar Conv0 mismatches | scalar Act0 mismatches | scalar Conv1 mismatches | IME status |
| --- | ---: | ---: | ---: | ---: | ---: |
| `synthetic_seeded` | `0` | `0` | `0` | `0` | `1` |
| `synthetic_gradient` | `0` | `0` | `0` | `0` | `1` |

Host IME status `1` means `not built with IME`, as expected for the host-native scalar-only build.

## Board

Board target: `svt@banana`

Board identity log:

```text
Linux bf3 6.6.63 #2.2.7.2 SMP PREEMPT Fri Aug 15 12:32:44 UTC 2025 riscv64
online CPUs: 0-7
```

Only CPU0-3 were used for IME tests.

| CPU | case | scalar status | scalar Conv0 mismatches | scalar Act0 mismatches | scalar Conv1 mismatches | IME status | IME Conv0 mismatches | IME Act0 mismatches | IME Conv1 mismatches |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `synthetic_seeded` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| 0 | `synthetic_gradient` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| 1 | `synthetic_seeded` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| 1 | `synthetic_gradient` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| 2 | `synthetic_seeded` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| 2 | `synthetic_gradient` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| 3 | `synthetic_seeded` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| 3 | `synthetic_gradient` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |

Stage 6 did not run IME on CPU4-7.

