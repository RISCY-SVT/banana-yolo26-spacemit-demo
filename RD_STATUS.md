# YOLO26 R&D Status

## Current Stage

`BANANA-YOLO26-SPACEMIT-DEMO-RD-BOOTSTRAP-001`

This is a bootstrap and forensic/API-discovery stage. It does not establish a
production demo path.

## Guardrails

- Do not modify `/data/banana-yolo11-spacemit-demo`.
- Do not push from this repository until a dedicated YOLO26 remote is approved.
- Keep downloaded runtimes, model checkpoints, generated ONNX files, raw logs,
  and probe binaries out of git unless explicitly promoted.
- Preserve exact provenance and SHA256 for every runtime/model artifact.

## Bootstrap Questions

- Which SpacemiT ORT 2.0.4 assets are available and usable on K1X?
- What provider options and architecture-selection strings exist in 2.0.4?
- Does 2.0.4 run on K1X without SIGILL or ABI/toolchain changes?
- What is the YOLO26n ONNX output contract for `end2end=True` and
  `end2end=False`?
- Does CPU ONNX decode match Ultralytics before EP acceleration?
- Is INT8 feasible after float semantics are correct?
