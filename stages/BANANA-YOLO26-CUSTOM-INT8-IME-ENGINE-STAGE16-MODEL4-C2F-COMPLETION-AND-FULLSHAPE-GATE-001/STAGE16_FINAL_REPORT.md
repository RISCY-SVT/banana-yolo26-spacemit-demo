# Stage 16 Final Report

classification: `stage16-model4-c2f-compact-correct-fullshape-gate-proven-conv-dominates`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE16-MODEL4-C2F-COMPLETION-AND-FULLSHAPE-GATE-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `14d0e74affce5abbb0667f9b759972b56ccb5b2b`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
graph_wide_scheduler_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
sliding_vmadot123_used: false
vmadotn_used: false
selected_subset: `candidate_I_model4_split_first_branch_fullshape_gate` and `candidate_J_model4_c2f_complete_compact`

## Summary

Stage16 first ran the required representative/full-shape gate for the Stage15 model4 branch-entry subset. The gate passed with real model4 branch-entry dimensions (`80x80x64 -> Split_output_1 80x80x32 -> /model.4/m.0/cv1/conv 80x80x16`) and `mismatches=0`.

After that gate, Stage16 completed a compact `/model.4` C2f-style candidate through float-domain Add, float-domain Concat, post-Concat Q/DQ, and `/model.4/cv2/conv/Conv`. This compact candidate also passed host and board correctness with `mismatches=0`.

No full YOLO26 inference, graph-wide scheduler, full-image/camera path, COCO/mAP, production claim, default backend switch, `/data/ncnn` mutation, XSlim, or sliding-vmadot implementation was done.

## Stage16A Gate

| candidate | shape_class | total_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct | mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| `stage16A_IME_A2_rvv_f32_lut` | `full_shape_model4_branch_entry` | 25491.0 | 79.8539 | 19.4265 | 0.711448 | 0 | 0 |

Decision: full-shape/representative branch-entry timing is proven and Conv-dominated. This is selected-subset evidence only and not model FPS.

## Model4 Compact C2f

| candidate | shape_class | total_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct | mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| `stage16_IME_A2_rvv_f32_lut` | `compact` | 197.493 | 65.3146 | 14.7337 | 11.6465 | 0.210978 | 0 |

Add status: `pass-float-domain-compact`
Concat status: `pass-float-domain-compact`
Post-Concat Q/DQ status: `pass-compact`
Model4 cv2 Conv status: `pass-compact`

## Validation

- Stage15 replay: pass
- Host build: pass
- Host CTest: pass, `33/33`
- RISC-V cross build with `Y26_K1X_ENABLE_IME=ON`: pass
- Board CPU0/1/2/3 Stage15 correctness: pass
- Board CPU0/1/2/3 Stage16 compact correctness: pass
- Board CPU0 Stage16A representative/full-shape microbench: pass
- `git diff --check`: pass
- symlink scan: pass
- changed-file secret-like scan: pass

## Bottleneck Decision

Representative/full-shape Stage16A is Conv-dominated (`conv_share_pct=79.8539`). Further compact-only graph expansion is not recommended until a Conv/IME roofline and controlled cluster0 threading feasibility stage is run on representative/full-shape model4 boundaries.

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE17-CONV-IME-ROOFLINE-AND-CLUSTER0-THREADING-FEASIBILITY-001`
