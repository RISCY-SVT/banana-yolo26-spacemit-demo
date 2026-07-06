# Selected Repair Timing Report

protocol: `warmup=10 runs=100 repeats=5`
board_affinity: `taskset -c 0-3`

| candidate | mean_total_us | stddev_total_us | mean_merge_us | merge_share_pct | mismatches |
|---|---:|---:|---:|---:|---:|
| B1_threaded_branch0_4t | 149539 | 76.3189 | 66564.3 | 44.513 | 0 |
| C2_split0_concat_lut_4t | 116338 | 121.933 | 29791.6 | 25.6078 | 0 |

total_delta_us: `-33201`
merge_delta_us: `-36772.7`
total_speedup_vs_B1_4t: `1.2854x`

This is selected-subset microbench evidence only.
