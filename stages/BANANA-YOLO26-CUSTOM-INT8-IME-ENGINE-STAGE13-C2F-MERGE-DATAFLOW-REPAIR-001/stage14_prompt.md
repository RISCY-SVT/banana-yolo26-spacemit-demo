# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001

User-facing summaries must be in Russian. Code, comments, commands, paths,
identifiers, report filenames, and artifact names stay in English.

## Mission

Expand from Stage 13 `candidate_G_model2_c2f_add_concat_cv2_conv` to the next
bounded YOLO26 backbone/C2f subset using the accepted Stage 13 `A2_fused_qdq_nhwc`
merge dataflow.

Do not implement full YOLO26 inference, a graph-wide scheduler, camera/full-image
demo, COCO/mAP, model FPS, production claims, XSlim, `/data/ncnn` mutation, or
sliding `vmadot1/2/3`.

## Required Start

- Expected previous stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE13-C2F-MERGE-DATAFLOW-REPAIR-001`
- Read Stage 13 final reports, candidate matrix, component timing, and source hygiene report.
- Replay Stage 13 A2 correctness on CPU0/1/2/3.
- Replay Stage 13 A2 CPU0 microbench.

## Technical Guardrails

- Use plain `smt.vmadot` MMT4D Conv path only.
- Use accepted `A2_rvv_f32_lut` activation path.
- Use accepted `A2_fused_qdq_nhwc` C2f merge dataflow.
- Keep Add/Concat float-domain unless a separate oracle proves otherwise.
- Keep timing buckets non-overlapping.
- Treat merge/QDQ, activation/requant, Conv, correction, and pack/layout as first-class buckets.

## Candidate Scope

Select the next bounded block only if ONNX CPU oracle, tensor contracts, Q/DQ
scales, zero-points, branch layout, and output boundary are clear. Stop at the
largest safe subset; do not force branches or Concat if ambiguous.

## Reports

Create Stage 14 reports for subset selection, tensor contracts, oracle, block
correctness, component timing, source hygiene, and Stage 15 prompt.
