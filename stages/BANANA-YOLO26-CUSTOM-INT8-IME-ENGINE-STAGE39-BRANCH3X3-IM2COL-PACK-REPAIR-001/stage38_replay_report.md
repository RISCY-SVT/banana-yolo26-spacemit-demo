# Stage38 Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


Replay command log: `/data/ncnn-logs/ai-team/2026-07-09_10-31-09/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001/run_logs/board_stage38_replay.log`
Replay-after command log: `/data/ncnn-logs/ai-team/2026-07-09_10-31-09/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001/run_logs/board_stage38_replay_after.log`

## Correctness

- mismatches: `0`
- max_abs_diff: `0`
- output_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- affinity_ok: `1`
- attribution_pct: `99.901`
- FRM sweep: pass for RNE/RTZ/RDN/RUP/RMM

## Stable Timing

| run | mean_total_us | stddev_total_us | CV % |
|---|---:|---:|---:|
| Stage38 replay before candidate | 30334.500 | 207.007 | 0.682 |
| Stage38 replay after candidate | 30206.400 | 264.204 | 0.875 |

Same-session replay is stable enough for A/B comparison; the after-candidate replay stayed within `128.100 us` of the first replay.
