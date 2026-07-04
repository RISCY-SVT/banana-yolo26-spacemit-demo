# Stage 9 Baseline Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`
board: `svt@banana`
affinity: `taskset -c 0`
source_binary: `bench_stage9_activation_fusion`

## Replay Results

| mode | total_us | activation_us | activation_share | conv0_us | conv1_us | conv2_us | mismatches | checksum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_int8_lut | 347932 | 192766 | 55.4034% | 68211.9 | 64052 | 21767.4 | 0 | 707794080 |
| A2_rvv_f32_lut | 182746 | 27426.9 | 15.0082% | 68061.8 | 64207 | 21898.7 | 0 | 707794080 |

## Gate

A2 baseline replay passed: `mismatches=0`, activation share `15.0082%`, below the Stage 10 continuation gate of `20%`.

## Caveat

This is selected-subset microbench evidence only, not YOLO26 full-model FPS.
