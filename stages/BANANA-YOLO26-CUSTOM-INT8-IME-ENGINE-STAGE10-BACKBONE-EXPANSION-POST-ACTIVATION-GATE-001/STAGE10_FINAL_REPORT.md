# Stage 10 Final Report

classification: `stage10-backbone-expanded-ready-for-branch-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `2eddbdf3418423e796a24ba060b5cf059a0f7e48`
end_head: `56a612bfcede03d626811cf6e4be29f13bfbdb2c`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false
selected_subset: `candidate_E_branch1_stage9_split_model2_m0_cv1_conv`
selected_mode: `A2_rvv_f32_lut`

## Proven

- Stage 9 A2 baseline replay passed with activation share `15.0594%` on the old subset.
- RVV A2 now uses explicit RNE conversion and passed ambient-FRM regression on CPU0-3.
- Stage 10 crossed `/model.2/Split` and ran the first branch Conv `/model.2/m.0/cv1/conv/Conv`.
- New boundary LUT matched ONNX Runtime 256-code oracle with `mismatches=0`.
- Host CTest passed: `27/27`.
- RISC-V cross build passed.
- Board CPU0/1/2/3 correctness passed with `mismatches=0`.
- Stage 10 expanded subset A2 microbench: total `234341 us`, activation share `15.379%`, pack/layout share `0.482372%`.

## Broken

- Nothing blocking in the selected Stage 10 scope.

## Unknown

- Full YOLO26 inference correctness or speed.
- COCO/mAP impact.
- Full-image/camera behavior.
- Later branch Add/Concat contracts and bottlenecks.

## Bottleneck

The new dominant bucket is `conv / IME`, not activation/requant, Split copy, or pack/layout.

## Next

Recommended next stage after review/approval:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE11-BRANCH-BLOCK-EXPANSION-001`
