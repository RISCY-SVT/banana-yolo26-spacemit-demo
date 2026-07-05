# Stage 16A Full-Shape Gate Report

selected_subset: `candidate_I_model4_split_first_branch`
shape_class: `full_shape_model4_branch_entry`
full_shape_timing_status: `proven-for-branch-entry`
representative_shape_timing_status: `proven`
compact_only: false

The gate uses real `/model.4` branch-entry tensor dimensions and quantization contracts: `/model.4/cv1` output `80x80x64`, `/model.4/Split_output_1` `80x80x32`, and `/model.4/m.0/cv1/conv/Conv` output `80x80x16`.

Input values are a tiled real compact accumulator pattern from Stage14 fixtures. This preserves real channel counts, spatial dimensions, scale/zp contracts, and memory layout without claiming full graph execution or full model FPS.

| candidate | total_us | conv_us | activation_requant_us | split_us | post_qdq_us | correction_us | conv_share_pct | activation_share_pct | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | 92597.2 | 52479.1 | 39934.0 | 182.48 | 31941.7 | 203.727 | 56.6746 | 43.1266 | 0 |
| `stage16A_IME_A2_rvv_f32_lut` | 25491.0 | 20355.5 | 4951.99 | 181.355 | 3712.22 | 197.77 | 79.8539 | 19.4265 | 0 |

Decision: Stage16A passes correctness and proves non-compact branch-entry timing. The representative/full-shape branch-entry is Conv-dominated (`conv_share_pct=79.8539`) and activation is below the Stage16 stop threshold (`activation_share_pct=19.4265`).

Caveat: this is selected-subset branch-entry timing only. It is not full YOLO26 inference timing or model FPS.
