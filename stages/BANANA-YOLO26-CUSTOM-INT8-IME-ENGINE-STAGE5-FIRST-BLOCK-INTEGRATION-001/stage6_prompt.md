# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE6-MULTI-BLOCK-BACKBONE-SUBSET-001

You are Codex working in the Banana-Pi BPI-F3 / SpacemiT K1X / X60 / YOLO26 custom INT8 IME `smt.vmadot` engine project.

User-facing summaries must be in Russian. Code, comments, identifiers, commands, paths, report filenames, and artifact names stay in English.

## Mission

Extend the Stage 5 first-block runner from `block0_conv_only` to a small multi-block backbone subset, while preserving exact CPU oracle comparison, persistent prepack/workspace dataflow, and cluster0-only IME execution.

## Start State

Required predecessor:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001
classification: stage5-first-block-ready-for-multiblock-stage
```

Stage 5 proved:

- `/model.0/conv/Conv` selected as `block0_conv_only`
- ONNX CPU oracle generated for deterministic inputs
- host CTest passed
- board CPU0/1/2/3 correctness passed
- board block microbench: scalar `463480 us`, IME total packing included `71932.7 us`
- no full YOLO26 engine, no full graph scheduler, no model FPS, no camera, no COCO/mAP

## Scope

Allowed:

- integrate the next tractable directly connected Conv/activation boundary only if Q/DQ metadata is clear
- create a multi-block CPU oracle with ONNX Runtime tooling only
- keep weights prepacked once and workspaces reused
- run board cluster0 correctness and block-only microbench

Forbidden:

- full YOLO26 inference
- graph-wide scheduler
- camera/full-image demo
- COCO/mAP
- model FPS or production claims
- `/data/ncnn` mutation
- XSlim
- `vmadot1/2/3`, `vmadotn`, FP/vfmadot implementation
- CPU4-7 IME execution
- source push without explicit authorization

## Required Decision

If the SiLU boundary after `/model.0/conv/Conv` is still not tractable as an int8/local boundary, Stage 6 should select a clear Conv-to-Conv subset only after documenting the intervening Q/DQ and activation contract. If that is not tractable, stop and recommend a targeted activation/requant stage instead of expanding scheduler scope.
