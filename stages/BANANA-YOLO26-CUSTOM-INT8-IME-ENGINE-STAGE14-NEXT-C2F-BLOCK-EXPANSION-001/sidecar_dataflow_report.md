# Stage 14 Sidecar Dataflow Report

## Scope

The sidecar was limited to Stage 13 merge dataflow cleanup. It did not change the selected graph math, did not add a generic memory planner, and did not expand the graph.

## Attempted Candidate

`A2_post_concat_qdq_rvv` was implemented by vectorizing the Stage 13 post-Concat QDQ segment writer in `custom_int8_engine/src/c2f_block_runner.cpp` when `__riscv_vector` is available.

Properties:

- Uses explicit RNE conversion: `__riscv_vfcvt_x_f_v_i32m4_rm(..., __RISCV_FRM_RNE, ...)`.
- Keeps scalar fallback for host-native and non-RVV builds.
- Preserves `concat_mismatches=0` and `model2_cv2_mismatches=0`.
- Does not use `vmadot1/2/3`, `vmadotn`, FP/vfmadot, XSlim, or ncnn source changes.

## Decision

`A2_post_concat_qdq_rvv` is accepted as the Stage 14 sidecar default because it keeps the Stage 13 accepted `A2_fused_qdq_nhwc` dataflow while reducing replay merge cost versus the Stage 13 final report.

`A1_fused_add_concat` measured slightly lower total time in one CPU0 replay (`433682 us` vs `436117 us`), but it is not selected for Stage 14 expansion because the already accepted A2 path is a cleaner handoff for the next block and removes the float Concat materialization.
