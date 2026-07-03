# Stage 4 Final Report

classification: `stage4-packing-repaired-ready-for-first-block-integration`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `2d0fd778619aa14189921905d1ba36afc11102ff`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false

## Result

Stage 4 repaired the selected real Conv packing/dataflow path enough to recommend first-block integration next.

The implementation still uses only plain `smt.vmadot` MMT4D `4x4x8 s8xs8->s32`. No `vmadot1/2/3`, `vmadotn`, FP/vfmadot, full YOLO26 engine, graph scheduler, camera, COCO/mAP, or model-level FPS work was done.

## Proven

- Stage 3 baseline was reproduced before modification.
- Host-native CTest after Stage 4 changes: `16/16` pass.
- RISC-V cross build with `Y26_K1X_ENABLE_IME=ON`: pass.
- Board cluster0 smoke: pass.
- Real Conv1x1 selected-node fixture: pass.
- Real Conv3x3 selected-node fixture: pass.
- Stage 4 persistent prepack/workspace M-major and N-major paths: pass.
- Conv1x1 packing-included M-major path improved from Stage 3 `46649.6 us` to `21843.2 us`.
- Conv3x3 packing-included M-major path improved from Stage 3 `149121 us` to `37097.9 us`.

## Broken / Residual

- Conv3x3 `packA` remains the largest measured component (`24951.5 us` of `37097.9 us` raw M-major path), but it is no longer blocking the selected real-node microbench.
- This is still selected-layer/block evidence only.

## Unknown

- Full YOLO26 inference speed.
- Full-image pipeline speed.
- COCO/mAP.
- Accuracy after integrating downstream graph blocks and full requant/dequant policy.

## Next

Recommended next stage:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001`
