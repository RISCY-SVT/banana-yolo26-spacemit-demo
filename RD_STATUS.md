# YOLO26 R&D Status

## Current Stage

`BANANA-YOLO26-INT8-RT204-OPERATOR-SUPPORT-FORENSICS-001`

This is an INT8 calibration/export and rt204 operator-support forensic stage.
It does not establish a production demo path and it does not modify the frozen
YOLO11 production repo.

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
- Does rt204 SpaceMIT EP compile and execute CPU-good Q/DQ INT8 models?
- Does rt204 deserve a future YOLO11 production adoption gate?

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
- Ultralytics `quantize=8` remains unsuitable for this checkpoint/graph: every
  tested Q/DQ export returned zero detections in the CPU oracle, even with
  task-local small and representative calibration YAML files.
- Manual ONNX Runtime static Q/DQ over `Conv` and `MatMul` creates CPU-good
  INT8 candidates for both end-to-end `[1,300,6]` and traditional `[1,84,8400]`
  contracts.
- RT204 SpaceMIT EP does not yet accept those CPU-good INT8 candidates: the EP
  compiler fails inside the Q/DQ/Conv subgraph with
  `output_type not implemented for clip minmax`.
- Provider filter diagnostics show that disabling `QuantizeLinear`,
  `DequantizeLinear`, and `Conv` can recover correctness through CPU-heavy
  fallback, but that path is slower and is not an accelerated INT8 solution.
- Frozen YOLO11 production models were probed read-only with rt204. The direct
  tensor probe did not abort for dynamic640 INT8, vendor320 q.onnx, or FP16
  keep_io 640 and produced sane semantics; this remains only a future separate
  adoption-gate signal.

Raw evidence for this stage:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
```
