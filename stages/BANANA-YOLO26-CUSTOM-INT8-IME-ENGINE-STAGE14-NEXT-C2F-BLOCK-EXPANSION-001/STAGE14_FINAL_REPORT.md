# Stage 14 Final Report

classification: `stage14-next-c2f-expanded-ready-for-next-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `9219f897a47d76e8b06031d29dcc18c498cf48a0`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false
selected_subset: `candidate_H3_model2_act_model3_act_model4_cv1_conv`

## Proven

- Stage 13 selected subset replay passed.
- Stage 13 traceability was fixed to `9219f897a47d76e8b06031d29dcc18c498cf48a0`.
- Stage 14 selected `candidate_H3_model2_act_model3_act_model4_cv1_conv` and stopped before `/model.4/Split`.
- Boundary-specific ONNX Runtime LUT oracles for `/model.2/cv2/act` and `/model.3/act` passed with mismatches `0`.
- Host-native CTest passed: `31/31`.
- RISC-V cross build passed.
- Board CPU0/1/2/3 correctness passed for RNE, Stage13 replay, and Stage14 next-C2f runner.
- CPU0 compact selected-subset microbench passed with `mismatches=0`.

## Timing

CPU0 `bench_stage14_next_c2f 3`, compact deterministic fixture:

| selected path | total_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct |
|---|---:|---:|---:|---:|---:|
| `stage14_IME_A2_rvv_f32_lut` | `139.04` | `69.1314` | `16.0846` | `8.54121` | `0.339711` |

This is selected-subset microbench evidence only. It is not full YOLO26 inference, full-image speed, camera speed, COCO/mAP, or production readiness.

## Broken

- No full YOLO26 engine.
- No graph-wide scheduler.
- No default backend switch.
- No ncnn source mutation.
- No XSlim.
- No `vmadot1/2/3`, `vmadotn`, FP/vfmadot.
- No full-image/camera/COCO/mAP.

## Unknown

- Full-shape `/model.3` and `/model.4` block performance is not proven by compact fixtures.
- `/model.4/Split` and following branch contracts remain for Stage 15.
- Full-model accuracy and speed remain unknown.

## Next

Recommended next step:
`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001`.
