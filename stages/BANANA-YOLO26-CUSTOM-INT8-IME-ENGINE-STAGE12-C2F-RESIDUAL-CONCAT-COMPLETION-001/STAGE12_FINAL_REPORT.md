# Stage 12 Final Report

classification: `stage12-c2f-block-complete-ready-for-next-block-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE12-C2F-RESIDUAL-CONCAT-COMPLETION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `4cf60ff8568cccbfbb48056a962414aad66f3480`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false
selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`

## Proven

- Stage 11 traceability report was patched from placeholder to `4cf60ff8568cccbfbb48056a962414aad66f3480`.
- Stage 11 baseline replay passed on board CPU0/1/2/3.
- `/model.2/m.0/Add` is float-domain and implemented as explicit measured float fallback.
- `/model.2/Concat` is float-domain and implemented as explicit measured float materialization.
- Post-Concat Q/DQ matched a compact ONNX Runtime micro-oracle with `mismatches=0`.
- `/model.2/cv2/conv/Conv` corrected int32 output passed compact host and board fixtures.
- Host CTest passed: `29/29`.
- RISC-V cross build passed.
- Board CPU0/1/2/3 correctness passed with `mismatches=0`.

## Timing

CPU0 selected-subset microbench:

| path | total_us | activation_share_pct | conv_share_pct | add_concat_share_pct | pack_layout_share_pct |
|---|---:|---:|---:|---:|---:|
| `stage12_scalar_reference` | `1817490` | `18.0372` | `69.4162` | `5.30091` | `7.30878` |
| `stage12_scalar_A2_rvv_f32_lut` | `1548290` | `5.74612` | `80.0864` | `5.78266` | `8.43496` |
| `stage12_IME_A2_rvv_f32_lut` | `582039` | `15.1699` | `47.0785` | `15.4979` | `22.3855` |

## Broken

- No full YOLO26 engine.
- No graph-wide scheduler.
- No camera/full-image path.
- No COCO/mAP.
- No production/model FPS claim.

## Unknown

- Whether Stage 12 merge path can be fused enough to reduce `pack_layout_share_pct`.
- Later backbone/head correctness.
- Full model speed and accuracy.

## Decision

Stage 12 completes the first `/model.2` C2f-style block boundary and is ready
for a focused Stage 13 merge-dataflow repair before further graph expansion.
