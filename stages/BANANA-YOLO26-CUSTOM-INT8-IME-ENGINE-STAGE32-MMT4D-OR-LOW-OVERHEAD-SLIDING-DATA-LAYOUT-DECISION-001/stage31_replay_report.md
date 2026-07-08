# Stage31 Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001
source_stage: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE31-VMADOT123-DIRECT-CONV-INTEGRATION-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 00aa667b8770cd9e6c7a5cdd24ac2714bb1d52a9
log_dir: /data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001

## Command

```text
taskset -c 0-3 ./bench_stage31_vmadot123_direct_conv --warmup 10 --runs 100 --repeats 5
```

Raw log:

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/stage31_replay_board.log
```

## Correctness

```text
direct_status: 0
direct_mismatches: 0
direct_max_abs_diff: 0
mmt4d_1t_status: 0
mmt4d_1t_mismatches: 0
mmt4d_4t_status: 0
mmt4d_4t_mismatches: 0
checksum_direct: 1324192976
checksum_expected: 1324192976
affinity_1t: 1
affinity_4t: 1
```

## Stable Replay Table

| candidate | mean_us | stddev_us | cv_pct | mismatches |
|---|---:|---:|---:|---:|
| direct_sliding_vmadot123_stage31 | 58831.5 | 26.6264 | 0.0452588 | 0 |
| mmt4d_1thread_stage31_replay | 22141.0 | 105.014 | 0.474 | 0 |
| mmt4d_4thread_stage31_replay | 5916.17 | 56.8996 | 0.962 | 0 |

## Direct Sidecar Component Replay

| bucket | mean_us |
|---|---:|
| direct_total_us | 58831.5 |
| panel_build_us | 40512.2 |
| kernel_compute_us | 15872.5 |
| correction_us | 796.151 |
| writeback_us | 856.234 |

## Interpretation

Stage31 replay reproduced the important conclusion within same-session evidence: the direct/sliding sidecar is correct, but it is much slower than both same-thread and current best 4-thread MMT4D.

The direct sidecar remains dominated by panel build, not by `smt.vmadot1/2/3` compute.
