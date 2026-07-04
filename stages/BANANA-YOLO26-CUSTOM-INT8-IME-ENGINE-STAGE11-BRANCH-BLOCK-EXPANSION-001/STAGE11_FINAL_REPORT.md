# Stage 11 Final Report

classification: `stage11-branch-cv2-correct-add-deferred`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE11-BRANCH-BLOCK-EXPANSION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `56a612bfcede03d626811cf6e4be29f13bfbdb2c`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false
selected_subset: `candidate_F_model2_m0_cv1_act_cv2_conv`
selected_mode: `A2_rvv_f32_lut`

## Proven

- Stage 10 replay passed before and after Stage 11 changes.
- RVV RNE regression passed on CPU0-3.
- New boundary `/model.2/m.0/cv1/act/Mul` has a 256-code ONNX Runtime LUT oracle with `mismatches=0`.
- Stage 11 added `/model.2/m.0/cv2/conv/Conv` and proved corrected int32 output.
- Host CTest passed: `28/28`.
- RISC-V cross build passed.
- Board CPU0/1/2/3 correctness passed with `mismatches=0`.
- Stage 11 CPU0 microbench: IME A2 total `269372 us`, activation share `14.8755%`, conv share `84.4801%`.

## Broken

- Residual Add is not implemented in Stage 11.

## Unknown

- `/model.2/m.0/cv2` activation + residual Add + Concat correctness.
- `/model.2/cv2/conv/Conv` contract after Concat.
- Full YOLO26 correctness, full-image speed, COCO/mAP, camera behavior, and production readiness.

## Decision

Stage 11A is correct and board-proven. Stage 11B residual Add is deferred because the ONNX Add is float-domain before Concat Q/DQ. The next stage should complete the C2f residual/Concat contract before further Conv expansion.
