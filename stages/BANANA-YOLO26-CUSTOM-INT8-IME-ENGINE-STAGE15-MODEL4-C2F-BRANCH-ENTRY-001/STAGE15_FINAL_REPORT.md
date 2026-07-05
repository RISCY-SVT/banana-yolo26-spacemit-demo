# Stage 15 Final Report

classification: `stage15-model4-branch-correct-but-fullshape-unproven`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `5cc09059f83eaef6af8c9a6aee3eab1e4edd46e7`
end_head: `14d0e74affce5abbb0667f9b759972b56ccb5b2b`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false
threading_default_enabled: false
selected_subset: `candidate_I_model4_split_first_branch`
stage14_baseline_rechecked: `pass`
stage14_timing_caveat_recorded: `yes`
model4_split_status: `pass`
model4_branch0_status: `pass`
block_oracle: `pass`
host_tests: `pass`
board_tests: `pass`
full_shape_timing_status: `not_proven`
microbench_done: `yes`
result_packet: `/exchange/results/outbox/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001`
log_dir: `/data/ncnn-logs/ai-team/2026-07-05_13-30-42/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001`
next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE16-MODEL4-C2F-COMPLETION-AND-FULLSHAPE-GATE-001 after review/approval`

## Proven

- Stage 14 baseline replay passed on host and board CPU0/1/2/3.
- Stage 14 compact timing caveat was recorded: `139.04 us` is compact selected-subset evidence only.
- Stage 15 selected `/model.4/Split_output_1` plus `/model.4/m.0/cv1/conv/Conv` and branch activation/QDQ.
- `/model.4/Split_output_1` contract was documented as float Split followed by Q/DQ only on output1.
- Boundary-specific ONNX Runtime 256-code LUT oracles passed with `mismatches=0` and `max_abs_diff_u8=0`.
- Host-native CTest passed: `32/32`.
- RISC-V cross build passed with `Y26_K1X_ENABLE_IME=ON`.
- Board CPU0/1/2/3 correctness passed with `mismatches=0`.
- CPU0 compact selected-subset microbench passed with `mismatches=0`.

## Compact Timing

`taskset -c 0 ./bench_stage15_model4_branch 3`

| candidate | total_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct |
|---|---:|---:|---:|---:|---:|
| `stage15_IME_A2_rvv_f32_lut` | `160.038` | `64.2282` | `20.4637` | `7.81066` | `0.190788` |

This is compact selected-subset evidence only. It is not full-shape timing and not model FPS.

## Broken

- Full YOLO26 inference is not implemented.
- Graph-wide scheduler is not implemented.
- Full-image/camera/COCO/mAP was not run.
- Full-shape Stage15 timing is not proven.
- `/model.4/m.0/cv2`, `/model.4/m.0/Add`, `/model.4/Concat`, and `/model.4/cv2/conv` are deferred.
- No ncnn integration or default backend switch.

## Unknown

- Representative/full-shape performance for `/model.4` branch-entry remains unknown.
- Whether `/model.4` Add/Concat repeats Stage13 merge bottlenecks remains unknown.
- Whether Conv becomes dominant on representative/full-shape data remains unknown.

## Files Created/Modified

- `custom_int8_engine/include/y26_k1x_model4_branch_runner.h`
- `custom_int8_engine/src/model4_branch_runner.cpp`
- `custom_int8_engine/tests/stage15_model4_branch_fixture.h`
- `custom_int8_engine/tests/test_stage15_model4_branch_runner.cpp`
- `custom_int8_engine/tools/bench_stage15_model4_branch.cpp`
- `custom_int8_engine/tools/extract_stage15_model4_branch_oracle.py`
- `custom_int8_engine/CMakeLists.txt`
- `custom_int8_engine/tests/CMakeLists.txt`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001/STAGE14_FINAL_REPORT.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001/*`

## Human Decisions Needed

- Review whether Stage 16 should prioritize representative/full-shape selected-subset timing before completing `/model.4` C2f.
- Review whether compact timing remains acceptable only as correctness/shape smoke evidence.
