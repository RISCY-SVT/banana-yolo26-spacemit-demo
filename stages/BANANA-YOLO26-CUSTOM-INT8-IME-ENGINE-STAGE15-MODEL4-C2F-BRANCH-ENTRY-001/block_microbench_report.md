# Stage 15 Block Microbench Report

selected_subset: `candidate_I_model4_split_first_branch`
fixture: `synthetic_seeded`
iterations: `3`
board_cpu_affinity: `taskset -c 0`

Command:

`taskset -c 0 ./bench_stage15_model4_branch 3`

## Result

| candidate | correctness_status | status | mismatches | checksum | total_us | conv_us | activation_requant_us | split_us | merge_us | pack_layout_us |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scalar_reference_int8_lut` | `pass` | `0` | `0` | `333333` | `382.713` | `271.812` | `84.0947` | `9.99967` | `13.291` | `0.305667` |
| `stage15_IME_A2_rvv_f32_lut` | `pass` | `0` | `0` | `333333` | `160.038` | `102.789` | `32.7497` | `9.597` | `12.5` | `0.305333` |

This is selected-subset compact microbench evidence only.

It is not:

- full YOLO26 inference,
- full-shape selected-subset timing,
- full-image speed,
- camera speed,
- COCO/mAP,
- production readiness.

microbench_done: `yes`
