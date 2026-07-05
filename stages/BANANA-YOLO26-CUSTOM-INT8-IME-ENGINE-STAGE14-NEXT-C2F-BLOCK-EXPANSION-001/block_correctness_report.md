# Stage 14 Block Correctness Report

selected_subset: `candidate_H3_model2_act_model3_act_model4_cv1_conv`
fixture_header: `custom_int8_engine/tests/stage14_next_c2f_fixture.h`
fixture_sha256: `c8d46402b3858357c61945dbf0bcb60c4841f6948c8da1145e74cd88cfea69e7`

## Host

Host-native CTest passed: `31/31`.

`test_stage14_next_c2f_runner` checks:

- `/model.3/conv/Conv` input signed int8 handoff.
- corrected int32 output of `/model.3/conv/Conv`.
- `/model.4/cv1/conv/Conv` input signed int8 handoff.
- corrected int32 output of `/model.4/cv1/conv/Conv`.

## Board

Board CPU0/1/2/3 correctness passed with `mismatches=0` for:

- `test_stage10_rvv_rounding_control`
- `test_stage13_merge_dataflow`
- `test_stage14_next_c2f_runner`

## CPU0 Test Output Summary

| fixture | mode | model3_input | model3 | model4_input | model4_cv1 |
|---|---|---:|---:|---:|---:|
| `synthetic_seeded` | `ime_rvv_f32_lut` | `0` | `0` | `0` | `0` |
| `synthetic_gradient` | `ime_rvv_f32_lut` | `0` | `0` | `0` | `0` |

The board log is preserved in:
`/data/ncnn-logs/ai-team/2026-07-05_07-43-30/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001/run_logs/board_stage14_final.log`.
