# Component Timing Report

scope: selected-subset microbench only
board: `svt@banana`
CPU affinity: `taskset -c 0`

## Full Selected Subset

| path | total_us | conv0_us | act0_us | conv1_us | act1_us | conv2_us | activation_total_us | activation_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 `int8_lut` | 350531 | 69995.6 | 128878 | 64505.8 | 64007.3 | 21964.5 | 192885 | 55.0266% |
| A1 scalar-unrolled | 306914 | 70093.6 | 100084 | 64599.6 | 48919.9 | 22078 | 149004 | 48.5492% |
| A2 RVV f32 | 182420 | 70112.2 | 17555.1 | 64643.3 | 6916.17 | 22055.3 | 24471.3 | 13.4148% |
| A3 fixed-requant | 215074 | 69961.8 | 37850.3 | 64814.7 | 19353.5 | 21955.7 | 57203.8 | 26.5972% |
| A4 fused current-layout | 304990 | 69970.6 | 97947 | 64514 | 49000.7 | 22360.1 | 146948 | 48.1811% |

## Profile Sub-buckets

| path | act0_requant_arithmetic_us | act0_lut_store_us | act1_requant_arithmetic_us | act1_lut_store_us | activation_total_us | mismatches |
|---|---:|---:|---:|---:|---:|---:|
| A1 scalar-unrolled profile | 94086.4 | 1453.55 | 46957.2 | 769.904 | 143270 | 0 |
| A3 fixed-requant profile | 38013 | 1779.45 | 19661.2 | 776.155 | 60232.1 | 0 |

## Gates

| gate | threshold | A2 result | status |
|---|---:|---:|---|
| minimum activation share | `<40%` | `13.4148%` | pass |
| minimum total | `<=280000 us` | `182420 us` | pass |
| good activation share | `<30%` | `13.4148%` | pass |
| good total | `<=240000 us` | `182420 us` | pass |
| excellent activation share | `<20%` | `13.4148%` | pass |
| excellent total | `<=210000 us` | `182420 us` | pass |
