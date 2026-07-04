# Component Timing Report

scope: selected-subset microbench only, not YOLO26 full inference FPS
board: `svt@banana`
affinity: `taskset -c 0`
iterations: `1`

## Stage 9 Replay

| mode | total_us | activation_us | activation_share | conv0_us | conv1_us | conv2_us | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_int8_lut | 350401 | 195618 | 55.8268% | 68421.6 | 63764.7 | 21460.1 | 0 |
| A2_rvv_f32_lut | 182491 | 27481.9 | 15.0594% | 68479.1 | 63751.6 | 21640.5 | 0 |

## Stage 10 Expanded Subset

| mode | total_us | activation_us | activation_share | conv0_us | conv1_us | conv2_us | act2_us | split_us | branch_conv_us | pack_layout_share | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar_reference | 1137540 | 258265 | 22.7038% | 446324 | 275883 | 66008.4 | 65354.1 | 1130.64 | 89630.5 | 0.0993937% | 0 |
| A2_rvv_f32_lut IME | 234341 | 36039.4 | 15.379% | 68655.7 | 64146.1 | 25875.6 | 8597.32 | 1130.39 | 38137.2 | 0.482372% | 0 |

## Shares for Selected Stage 10 A2 Path

- activation_share: `15.379%`
- conv_share: `83.9864%`
- pack_layout_share: `0.482372%`
- split_branch_share: `16.7566%`

## Decision

Activation did not regress above the 40% gate. Pack/layout and Split copy are not dominant. The new dominant bucket is Conv/IME work, especially the added branch Conv.
