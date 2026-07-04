# Block Microbench Report

selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`
board_cpu: `CPU0`
command: `taskset -c 0 ./bench_stage12_c2f_block 3`
claim_scope: selected-subset microbench only

| path | total_us | mismatches | activation_share_pct | conv_share_pct | add_concat_share_pct | pack_layout_share_pct |
|---|---:|---:|---:|---:|---:|---:|
| `stage12_scalar_reference` | `1817490` | `0` | `18.0372` | `69.4162` | `5.30091` | `7.30878` |
| `stage12_scalar_A2_rvv_f32_lut` | `1548290` | `0` | `5.74612` | `80.0864` | `5.78266` | `8.43496` |
| `stage12_IME_A2_rvv_f32_lut` | `582039` | `0` | `15.1699` | `47.0785` | `15.4979` | `22.3855` |

Stage 11 replay IME A2 for comparison:

- total_us: `263171`
- activation_share_pct: `14.9207`
- conv_share_pct: `84.5192`

Do not compare these numbers as model FPS.
