# Model4 Correctness Report

selected_subset: `candidate_J_model4_c2f_complete_compact`
status: `pass`

Host CTest includes `test_stage16_model4_c2f_runner` and passed as part of `33/33` tests.

Board CPU0/1/2/3 correctness passed with `mismatches=0` for Stage16 compact C2f completion.

CPU0 compact bench correctness:

| candidate | status | mismatches | checksum |
|---|---:|---:|---:|
| `scalar_reference_int8_lut` | 0 | 0 | -143848 |
| `stage16_IME_A2_rvv_f32_lut` | 0 | 0 | -143848 |
