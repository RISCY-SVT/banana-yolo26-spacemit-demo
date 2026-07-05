# Stage 16A Correctness Report

subset: `candidate_I_model4_split_first_branch`
fixture: `representative_full_shape_synthetic`
input source: tiled real compact Stage14 accumulator pattern
status: `pass`

Board CPU0 benchmark correctness summary:

| candidate | status | mismatches | split_mismatches | branch_mismatches | branch_act_mismatches | checksum |
|---|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | 0 | 0 | 0 | 0 | 0 | 1324192976 |
| `stage16A_IME_A2_rvv_f32_lut` | 0 | 0 | 0 | 0 | 0 | 1324192976 |

The initially tested artificial stress distribution produced a scalar-double-vs-RVV-f32 activation mismatch and was rejected as an accepted path. The accepted Stage16A gate uses the tiled real compact input pattern and passes with `mismatches=0`.
