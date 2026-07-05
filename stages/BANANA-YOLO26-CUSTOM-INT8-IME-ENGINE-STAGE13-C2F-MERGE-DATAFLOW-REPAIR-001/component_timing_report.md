# Component Timing Report

Final CPU0 board command:

`taskset -c 0 ./bench_stage13_merge_dataflow 3`

## Selected Candidate

`A2_fused_qdq_nhwc`

| component | us |
|---|---:|
| `total_us` | `502570` |
| `conv_us` | `271459` |
| `activation_requant_us` | `88080.7` |
| `split_copy_us` | `56452.7` |
| `add_compute_us` | `0` |
| `concat_materialize_us` | `0` |
| `post_concat_qdq_us` | `83421.1` |
| `pack_for_model2_cv2_us` | `0` |
| `layout_copy_us` | `0` |
| `correction_us` | `3373.44` |
| `model2_cv2_conv_us` | `48480.9` |
| `merge_total_us` | `139874` |
| `merge_share_pct` | `27.9607` |
| `pack_layout_share_pct` | `0.162724` |
| `activation_share_pct` | `17.6073` |
| `conv_share_pct` | `54.2644` |

This is selected-subset microbench evidence only, not YOLO26 model FPS.
