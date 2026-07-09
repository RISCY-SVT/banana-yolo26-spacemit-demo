# Selected Lane Benchmark Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


Stable protocol: `taskset -c 0-3`, warmup=10, runs=100, repeats=5.

## Instrumented Attribution Runs

| run | mean_total_us | stddev_us | CV % | combined branch im2col us | combined branch conv us |
|---|---:|---:|---:|---:|---:|
| Stage38 replay | 30334.500 | 207.007 | 0.682 | 5558.070 | 10291.260 |
| Stage39 fastpack | 28073.000 | 192.905 | 0.687 | 5355.820 | 8882.390 |

## No-Instrument Timing

| run | mean_total_us | stddev_us | CV % | combined branch conv us |
|---|---:|---:|---:|---:|
| Stage38 replay no measure | 30253.700 | 307.965 | 1.018 | 10335.520 |
| Stage39 fastpack no measure | 27977.900 | 245.398 | 0.877 | 8764.340 |

- selected_cut_total_speedup: `1.080558x` instrumented, `1.081343x` no-instrument.
- combined_branch3x3_conv_speedup: `1.158614x` instrumented, `1.179270x` no-instrument.
- combined_branch3x3_im2col_pack_speedup: `1.037763x`; this misses the Stage39 1.30x im2col gate.
