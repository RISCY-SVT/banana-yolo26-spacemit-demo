# Stage 15 Block Correctness Report

selected_subset: `candidate_I_model4_split_first_branch`
selected_mode: `A2_rvv_f32_lut` for board IME path
full_shape_stage15_timing: `not_proven`

## Host

Command:

`ctest --test-dir .deps/custom_int8_engine/build-host-native-stage15 --output-on-failure`

Result: `32/32` tests passed.

New test:

`test_stage15_model4_branch_runner`

Host mode:

- `scalar_int8_lut`
- exact compact fixture checks

## Board

Board target: `svt@banana`
board dir: `/home/svt/yolo26-custom-int8-stage15/2026-07-05_13-30-42/final`

CPU0/1/2/3 all passed:

- `test_stage10_rvv_rounding_control`
- `test_stage14_next_c2f_runner`
- `test_stage15_model4_branch_runner`

No CPU4-7 IME execution was run.

## CPU0 Stage 15 Correctness Excerpt

| fixture | mode | status | split1_mismatches | branch0_mismatches | branch0_act_mismatches |
|---|---|---:|---:|---:|---:|
| `synthetic_seeded` | `scalar_int8_lut` | `0` | `0` | `0` | `0` |
| `synthetic_seeded` | `rvv_f32_lut` | `0` | `0` | `0` | `0` |
| `synthetic_seeded` | `ime_rvv_f32_lut` | `0` | `0` | `0` | `0` |
| `synthetic_gradient` | `scalar_int8_lut` | `0` | `0` | `0` | `0` |
| `synthetic_gradient` | `rvv_f32_lut` | `0` | `0` | `0` | `0` |
| `synthetic_gradient` | `ime_rvv_f32_lut` | `0` | `0` | `0` | `0` |

## Output Boundaries Checked

- `/model.4/Split_output_1` signed int8 storage.
- `/model.4/m.0/cv1/conv/Conv` corrected int32 output.
- `/model.4/m.0/cv1/act/Mul_output_0` signed int8 storage.

block_oracle: `pass`
host_tests: `pass`
board_tests: `pass`
