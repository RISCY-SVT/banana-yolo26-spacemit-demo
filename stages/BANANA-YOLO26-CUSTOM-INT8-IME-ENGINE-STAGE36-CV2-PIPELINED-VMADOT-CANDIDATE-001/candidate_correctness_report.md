# Candidate Correctness Report

Expected output SHA:

`70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`

| candidate | status | mismatches | max_abs_diff | checksum | expected_checksum | affinity_ok | output_sha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0_branch1_add_lut | 0 | 0 | 0 | 106597930 | 106597930 | 1 | match |
| A1_cv2_pipelined4 | 0 | 0 | 0 | 106597930 | 106597930 | 1 | match |
| A2_cv2_pipelined6 | 0 | 0 | 0 | 106597930 | 106597930 | 1 | match |
| A1_cpu0_single_thread | 0 | 0 | 0 | 106597930 | 106597930 | 1 | match |
| A1_frm_sweep | 0 | 0 | 0 | 106597930 | 106597930 | 1 | match |

No CPU4-7 IME execution was used.

Raw evidence:

- `stage36_selected_cut_smoke.log`
- `stage36_stable_candidates.log`
- `stage36_cpu0_frm_selected.log`
