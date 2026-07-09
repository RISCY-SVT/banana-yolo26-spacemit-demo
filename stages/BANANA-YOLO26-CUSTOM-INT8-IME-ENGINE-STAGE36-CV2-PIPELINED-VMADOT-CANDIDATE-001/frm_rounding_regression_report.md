# FRM Rounding Regression Report

Candidate: `A1_cv2_pipelined4`

Command mode:

- selected real runner API through `bench_stage23_model4_runner_cut`
- `--merge-repair branch1_add_lut_cv2_pipelined4`
- `--frm-sweep`
- `taskset -c 0-3`

| ambient_frm | status | mismatches | max_abs_diff | after_frm | checksum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | 106597930 |
| 1 | 0 | 0 | 0 | 1 | 106597930 |
| 2 | 0 | 0 | 0 | 2 | 106597930 |
| 3 | 0 | 0 | 0 | 3 | 106597930 |
| 4 | 0 | 0 | 0 | 4 | 106597930 |

Result: pass. The accepted runner path restores the ambient FRM and preserves byte-exact output for RNE, RTZ, RDN, RUP, and RMM.
