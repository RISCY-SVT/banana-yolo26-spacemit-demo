# Stage 16A Component Timing Report

Non-overlapping timing buckets for the representative/full-shape branch-entry gate:

| candidate | total_us | conv_us | activation_requant_us | split_us | merge_us | post_qdq_us | pack_layout_us | correction_us | copy_us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | 92597.2 | 52479.1 | 39934.0 | 182.48 | 182.48 | 31941.7 | 0 | 203.727 | 0 |
| `stage16A_IME_A2_rvv_f32_lut` | 25491.0 | 20355.5 | 4951.99 | 181.355 | 181.355 | 3712.22 | 0 | 197.77 | 0 |

Dominant bucket for the selected IME path: `conv_us`.
