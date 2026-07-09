# Stage37 Replay Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001

## Replay Target

- mode: `Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4`
- output quantize mode: `rvv`
- affinity: `taskset -c 0-3`
- protocol: `warmup=10 runs=100 repeats=5`
- output SHA expected: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- raw log: `/data/ncnn-logs/ai-team/2026-07-09_08-54-07/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001/run_logs/board_stage37_replay.log`

## Correctness

- status: `pass`
- mismatches: `0`
- max_abs_diff: `0`
- output_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- affinity_ok: `1`
- CPU4-7 IME execution: `none`
- FRM sweep: `pass` for `RNE RTZ RDN RUP RMM`

## Stable Timing

| metric | value |
|---|---:|
| mean_total_us | 32890.5 |
| stddev_total_us | 269.396 |
| min_total_us | 32563.0 |
| max_total_us | 33301.4 |
| cv_total_pct | 0.819071 |
| mean_attribution_pct | 99.9173 |

## Buckets

| bucket | mean_us | share_pct |
|---|---:|---:|
| input_adapter | 2640.44 | 8.028 |
| conv | 18051.0 | 54.8823 |
| activation_requant | 3004.23 | 9.13404 |
| merge/post_concat_qdq | 2112.38 | 6.42248 |
| output_quantize | 7055.2 | 21.4506 |
| other_unattributed | 27.1973 | 0.0827 |

This is selected `/model.4` ONNX-cut timing only. It is not full YOLO26 inference, model FPS, camera/full-image performance, COCO/mAP, or production/default-backend readiness.
