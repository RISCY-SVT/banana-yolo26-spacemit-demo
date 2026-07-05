# Stage 13 Final Report

classification: `stage13-merge-dataflow-repaired-ready-for-next-block-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE13-C2F-MERGE-DATAFLOW-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `cae5301afc10a1ff2138335932d4939e3db64fc2`
end_head: `9219f897a47d76e8b06031d29dcc18c498cf48a0`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false
selected_subset: `candidate_G_model2_c2f_add_concat_cv2_conv`
best_candidate: `A2_fused_qdq_nhwc`

## Proven

- Stage 12 baseline replay passed on host and board.
- Stage 12 traceability placeholder was patched to `cae5301afc10a1ff2138335932d4939e3db64fc2`.
- Timing buckets were repaired so `split_copy_us` is no longer counted as `pack_layout_us`.
- `A1_fused_add_concat` removes the standalone `add_f32` materialization.
- `A2_fused_qdq_nhwc` removes standalone `add_f32` and `concat_f32`, writes post-Concat signed int8 NHWC storage directly, and reuses cached Split1 Q/DQ storage for exact dequantized split1 float values.
- CPU0/1/2/3 board correctness passed with `concat_mismatches=0` and `model2_cv2_mismatches=0`.
- Host CTest passed: `30/30`.
- RISC-V cross build passed.

## Timing

Final CPU0 selected-subset microbench, `iterations=3`:

| candidate | total_us | merge_total_us | merge_share_pct | pack_layout_share_pct | mismatches |
|---|---:|---:|---:|---:|---:|
| `A0_materialized_float_merge` | `580557` | `217677` | `37.643` | `0.140289` | `0` |
| `A1_fused_add_concat` | `506310` | `144517` | `28.6724` | `0.161052` | `0` |
| `A2_fused_qdq_nhwc` | `502570` | `139874` | `27.9607` | `0.162724` | `0` |

Stage 12 reported `pack_layout_share_pct=22.3855`, but Stage 13 shows that
figure was dominated by overlapping split/materialization accounting. Under
non-overlapping buckets, pack/layout is approximately `0.16%`; the real local
merge cost is split float materialization plus post-Concat Q/DQ.

## Broken

- No full YOLO26 engine.
- No graph-wide scheduler.
- No graph expansion beyond the Stage 12 selected subset.
- No camera/full-image path.
- No COCO/mAP.
- No model FPS or production claim.
- No `vmadot1/2/3`, `vmadotn`, FP/vfmadot, or XSlim use.

## Unknown

- Whether the next C2f block exposes larger branch/Concat costs.
- Whether future view-based packA from channel spans is worth adding once another block is selected.
- Full-model speed and accuracy remain unknown.

## Next

Recommended next step:
`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001`.

The Stage 13 repo-local report uses a pending end-head marker because a commit
hash cannot be embedded into the same commit that creates it. The task-run
final report and final response record the actual end head.
