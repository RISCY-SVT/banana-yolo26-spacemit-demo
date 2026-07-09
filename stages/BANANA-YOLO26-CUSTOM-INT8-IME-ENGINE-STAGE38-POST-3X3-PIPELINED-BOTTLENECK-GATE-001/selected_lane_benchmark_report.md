# Selected Lane Benchmark Report

## Protocol

- board: Banana-Pi BPI-F3 / SpacemiT K1X / X60
- affinity: `taskset -c 0-3`
- warmup: `10`
- runs: `100`
- repeats: `5`
- output SHA: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- baseline raw log: `/data/ncnn-logs/ai-team/2026-07-09_08-54-07/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001/run_logs/board_stage37_replay.log`
- candidate raw log: `/data/ncnn-logs/ai-team/2026-07-09_08-54-07/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001/run_logs/board_lane_a_rvv_direct.log`

## Stable Timing

| metric | Stage37 replay `rvv` | Stage38 Lane A `rvv_direct` | delta | speedup |
|---|---:|---:|---:|---:|
| mean_total_us | 32890.5 | 30341.5 | -2549.0 | 1.08401x |
| stddev_total_us | 269.396 | 281.576 | 12.180 | |
| cv_total_pct | 0.819071 | 0.928021 | 0.108950 | |
| output_quantize_us | 7055.2 | 4551.97 | -2503.23 | 1.54994x |
| conv_us | 18051.0 | 18048.1 | -2.9 | 1.00016x |
| activation_requant_us | 3004.23 | 3036.11 | 31.88 | 0.98950x |
| merge_us | 2112.38 | 2075.46 | -36.92 | 1.01779x |
| attribution_pct | 99.9173 | 99.9083 | -0.009 | |

## Decision

Lane A passed minimum gates:

- output_quantize speedup: `1.54994x` >= `1.30x`
- selected-cut total speedup: `1.08401x` >= `1.05x`

This is selected `/model.4` ONNX-cut timing only, not model FPS.
