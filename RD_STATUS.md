# YOLO26 R&D Status

## Current Stage

`BANANA-YOLO26-EXPORT-API-MISMATCH-FORENSICS-001`

This is an export/API mismatch forensic stage. It does not establish a
production demo path and it does not modify the frozen YOLO11 production repo.

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

## Current Answers

- Ultralytics `8.3.233` is not a suitable YOLO26 export/oracle path for this
  checkpoint. It exports traditional `[1,84,N]` by default, rejects
  `end2end=True/False`, and produces the false canonical `refrigerator` result.
- Ultralytics `8.4.0` fixes the default YOLO26 end-to-end behavior, producing
  `[1,300,6]`, but still rejects explicit `end2end=True/False`.
- Ultralytics `8.4.82` is the current preferred R&D oracle: default and
  `end2end=True` produce `[1,300,6]`; `end2end=False` produces traditional
  `[1,84,8400]`.
- Preprocess parity is proven for square letterbox input: explicit OpenCV
  letterbox hashes match Ultralytics `LetterBox` exactly on the oracle suite.
- ONNX Runtime CPU decode matches the latest PyTorch oracle for both
  end-to-end and traditional latest exports.
- RT204 SpaceMIT EP runs latest YOLO26 FP32 640 end-to-end and traditional
  exports on BPI-F3/K1X with sane semantic parity against CPU.
- INT8 is blocked: `quantize=8` exported a Q/DQ ONNX, but the generated model
  returned zero detections in host CPU smoke and used an unsuitable default
  `coco8` auto-calibration path.

Raw evidence for this stage:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
```
