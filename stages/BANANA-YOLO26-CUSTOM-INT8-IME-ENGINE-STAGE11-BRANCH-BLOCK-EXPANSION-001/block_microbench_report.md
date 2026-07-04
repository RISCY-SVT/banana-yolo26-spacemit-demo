# Block Microbench Report

scope: selected-subset microbench only, not full YOLO26 inference
board: `svt@banana`
affinity: `taskset -c 0`
iterations: `1`

## Stage 10 Replay

| path | total_us | activation_us | activation_share | conv_share | pack_layout_share | mismatches |
|---|---:|---:|---:|---:|---:|---:|
| Stage10 A2 | 234474 | 36026.9 | 15.365% | 83.9927% | 0.513195% | 0 |

## Stage 11A

| path | total_us | activation_us | activation_share | conv_share | pack_layout_share | mismatches |
|---|---:|---:|---:|---:|---:|---:|
| scalar reference | 1302620 | 275864 | 21.1777% | 78.69% | 0.0871781% | 0 |
| scalar A2 | 1066980 | 40526.6 | 3.79824% | 96.0394% | 0.10713% | 0 |
| IME A2 | 269372 | 40070.5 | 14.8755% | 84.4801% | 0.425471% | 0 |

Comparable Stage 11A IME speedup vs scalar reference: about `4.84x`.

This is not a model FPS result.
