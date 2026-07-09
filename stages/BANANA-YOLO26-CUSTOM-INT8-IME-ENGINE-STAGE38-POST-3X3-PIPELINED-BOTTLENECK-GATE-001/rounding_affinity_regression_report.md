# Rounding And Affinity Regression Report

## Stage37 Replay

| ambient_frm | status | mismatches | max_abs_diff | post_call_frm | checksum |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 106597930 |
| 1 | 0 | 0 | 0 | 1 | 106597930 |
| 2 | 0 | 0 | 0 | 2 | 106597930 |
| 3 | 0 | 0 | 0 | 3 | 106597930 |
| 4 | 0 | 0 | 0 | 4 | 106597930 |

## Stage38 Lane A Candidate

| ambient_frm | status | mismatches | max_abs_diff | post_call_frm | checksum |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 106597930 |
| 1 | 0 | 0 | 0 | 1 | 106597930 |
| 2 | 0 | 0 | 0 | 2 | 106597930 |
| 3 | 0 | 0 | 0 | 3 | 106597930 |
| 4 | 0 | 0 | 0 | 4 | 106597930 |

## Affinity

- board command used `taskset -c 0-3`.
- bench reported `affinity_ok=1` for replay and candidate.
- IME work remained cluster0-only, CPU0-3.
- CPU4-7 IME execution: `none`.
