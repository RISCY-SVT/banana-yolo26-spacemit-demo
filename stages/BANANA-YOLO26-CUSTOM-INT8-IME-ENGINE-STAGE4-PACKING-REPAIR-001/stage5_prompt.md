# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001

You are Codex working in the Banana-Pi BPI-F3 / SpacemiT K1X / X60 / YOLO26 custom INT8 IME engine project.

User-facing summaries must be in Russian. Code, commands, paths, identifiers, manifest keys, comments, filenames, and artifact names stay in English.

## Mission

Integrate the first selected YOLO26 real graph block using the Stage 4 repaired custom INT8 IME Conv dataflow.

Stage 5 may compose selected real Conv1x1/Conv3x3 nodes and local activation/requant boundaries, but must not implement full YOLO26 inference, camera, COCO/mAP, ncnn integration, production release artifacts, or model-level FPS claims.

## Required Start State

- Branch: `yolo26-custom-int8-engine`
- Stage 4 classification: `stage4-packing-repaired-ready-for-first-block-integration`
- Stage 4 commit should be present or Stage 4 changes should be validated before proceeding.
- Do not use XSlim.

## Implementation Foundation

- Use only plain `smt.vmadot` MMT4D `4x4x8 s8xs8->s32`.
- Use `Y26PrepackedConvWeights`.
- Use `Y26ConvWorkspace`.
- Use M-major loop order by default.
- Keep `vmadot1/2/3` documentation-only unless a later stage explicitly opens a direct-conv feasibility lane.
- Do not use `vmadotn` or FP/vfmadot.

## Scope

1. Select the first graph block around the Stage 3/4 real Conv nodes.
2. Build an ONNX CPU oracle for the block boundary only.
3. Prepack weights once for all block Conv nodes.
4. Reuse workspaces across block execution.
5. Implement only the needed local activation/requant/dequant pieces.
6. Compare block output against ONNX CPU oracle.
7. Run board cluster0 correctness and block-level microbench.

## Non-Goals

- No full YOLO26 inference.
- No full-image demo.
- No camera.
- No COCO/mAP.
- No ncnn source mutation.
- No `/data/ncnn` mutation.
- No production or headline FPS claims.

## Acceptance

- Selected block correctness passes on host oracle and board cluster0.
- Kernel/block timing separates prepack, packA, compute, correction/requant, and total.
- If timing regresses, classify and produce a repair plan rather than expanding scope.
