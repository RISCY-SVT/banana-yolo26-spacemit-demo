# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE13-C2F-MERGE-DATAFLOW-REPAIR-001

User-facing summaries must be in Russian. Code, comments, identifiers, commands,
paths, report filenames, and artifact names must be in English.

## Mission

Repair the Stage 12 C2f merge dataflow before expanding further.

Stage 12 completed:

```text
/model.2/m.0/Add
/model.2/Concat
post-Concat Q/DQ
/model.2/cv2/conv/Conv
```

Correctness passed on host and board CPU0-3, but the selected-subset timing shows
visible merge/layout cost:

```text
stage12_IME_A2_total_us: 582039
activation_share_pct: 15.1699
conv_share_pct: 47.0785
add_concat_share_pct: 15.4979
pack_layout_share_pct: 22.3855
```

## Scope

Optimize only the selected Stage 12 C2f merge dataflow. Do not expand to later
YOLO26 nodes until the Stage 12 merge path is characterized and repaired.

Allowed:

```text
- preserve exact Stage 12 oracle;
- fuse Split float materialization where safe;
- fuse Concat layout + post-Concat QDQ where safe;
- test direct write to signed int8 concat storage;
- keep /model.2/cv2/conv/Conv unchanged except for input layout handoff;
- run host tests and board CPU0-3 correctness/bench;
- report selected-subset timings only.
```

Forbidden:

```text
- full YOLO26 engine;
- graph-wide scheduler;
- COCO/mAP;
- camera/full-image demo;
- model FPS or production claims;
- /data/ncnn mutation;
- XSlim;
- vmadot1/2/3/vmadotn/FP/vfmadot implementation;
- push.
```

## Required Gates

1. Replay Stage 12 correctness and timing.
2. Preserve micro-ONNX Add+Concat+QDQ oracle with `mismatches=0`.
3. Implement a fused `Add+Concat+QDQ` candidate with no hidden approximation.
4. Compare:
   - Stage 12 materialized float merge;
   - fused merge;
   - scalar reference.
5. Board CPU0/1/2/3 correctness must pass with `mismatches=0`.

## Candidate Classification

Use one:

```text
stage13-c2f-merge-dataflow-repaired-ready-for-next-backbone-stage
stage13-correct-but-merge-still-dominates
stage13-blocked-oracle-mismatch
stage13-blocked-board-correctness
stage13-partial-needs-human-decision
```
