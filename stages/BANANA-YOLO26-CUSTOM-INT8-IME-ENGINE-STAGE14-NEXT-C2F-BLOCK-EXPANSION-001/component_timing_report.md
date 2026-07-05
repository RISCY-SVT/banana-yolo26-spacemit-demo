# Stage 14 Component Timing Report

All values are CPU0 selected-subset microbench values from `bench_stage14_next_c2f 3`.
This is compact selected-subset evidence, not YOLO26 full-model timing.

## Stage 14 Selected Path

| candidate | fixture | total_us | conv_us | activation_requant_us | merge_us | pack_layout_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct | mismatches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage14_IME_A2_rvv_f32_lut` | `synthetic_seeded` | `139.04` | `96.12` | `22.364` | `11.8757` | `0.472333` | `69.1314` | `16.0846` | `8.54121` | `0.339711` | `0` |

## Scalar Compact Reference

| candidate | fixture | total_us | conv_us | activation_requant_us | merge_us | pack_layout_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct | mismatches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | `synthetic_seeded` | `363.726` | `262.676` | `77.0757` | `13.39` | `0.389` | `72.2181` | `21.1906` | `3.68134` | `0.106949` | `0` |

## Interpretation

- `activation_share_pct` remains below the Stage 14 stop threshold.
- `pack_layout_share_pct` remains below the Stage 14 stop threshold.
- `merge_share_pct` remains below the Stage 14 stop threshold.
- Conv is the largest component, but stays just below the `>70%` diagnostic threshold for the selected IME candidate.
