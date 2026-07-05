# Stage 15 Component Timing Report

Timing source:

`taskset -c 0 ./bench_stage15_model4_branch 3`

This is compact selected-subset timing only. It is not full-shape performance and not model FPS.

## CPU0 Compact Timing

| candidate | total_us | conv_us | activation_requant_us | split_us | merge_us | add_us | concat_us | post_qdq_us | pack_layout_us | correction_us | branch0_conv_us | branch0_activation_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | `382.713` | `271.812` | `84.0947` | `9.99967` | `13.291` | `0` | `0` | `3.29133` | `0.305667` | `2.40267` | `8.56933` | `1.44433` | `71.0226` | `21.9733` | `3.47284` | `0.0798684` | `0` |
| `stage15_IME_A2_rvv_f32_lut` | `160.038` | `102.789` | `32.7497` | `9.597` | `12.5` | `0` | `0` | `2.903` | `0.305333` | `2.12433` | `7.41667` | `0.625` | `64.2282` | `20.4637` | `7.81066` | `0.190788` | `0` |

## Interpretation

- Compact Stage 15 IME timing is faster than compact scalar reference for the same fixture and boundary.
- Activation share remains below `40%` on compact evidence.
- Merge/share is Split-only in Stage 15 and remains below `35%` on compact evidence.
- Conv is the largest component but remains below the `>70%` diagnostic threshold for the compact IME candidate.
- Representative/full-shape timing was not proven in Stage 15.

full_shape_stage15_timing: `not_proven`
