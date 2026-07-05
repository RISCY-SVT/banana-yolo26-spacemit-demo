# Stage 13 Replay Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001`
replayed_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`
replayed_stage13_mode: `A2_fused_qdq_nhwc`
board_target: `svt@banana`
board_dir: `/home/svt/yolo26-custom-int8-stage14/2026-07-05_07-43-30/final`

## Gate Results

| gate | status | evidence |
|---|---|---|
| host CTest | pass | `31/31` in `final_host_build_ctest.log` |
| RISC-V cross build | pass | `final_cross_build.log` |
| CPU0-3 RNE regression | pass | `test_stage10_rvv_rounding_control`, mismatches `0` |
| CPU0-3 Stage13 correctness | pass | `test_stage13_merge_dataflow`, concat/model2_cv2 mismatches `0` |
| CPU0 Stage13 microbench | pass | `bench_stage13_merge_dataflow 3` |

## CPU0 Replay Timing

Final replay from `bench_stage13_merge_dataflow 3`:

| candidate | total_us | merge_total_us | conv_us | activation_requant_us | pack_layout_share_pct | mismatches |
|---|---:|---:|---:|---:|---:|---:|
| `A0_materialized_float_merge` | `503224` | `141522` | `269974` | `88580.2` | `0.16149` | `0` |
| `A1_fused_add_concat` | `433682` | `69763.9` | `271920` | `88882.4` | `0.182031` | `0` |
| `A2_fused_qdq_nhwc` | `436117` | `75406.4` | `269552` | `88069.9` | `0.185459` | `0` |

The Stage 13 accepted result was `A2_fused_qdq_nhwc` with `total_us=502570`.
The replay remains correct and within the expected R&D board timing variance.
