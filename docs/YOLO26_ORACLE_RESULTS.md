# YOLO26 Oracle Results

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
```

## Oracle Suite

The current oracle suite includes:

- canonical production photo;
- Ultralytics `bus.jpg`;
- Ultralytics `zidane.jpg`;
- Day 4 real camera still;
- blank white sanity image.

At `conf=0.25`, Ultralytics `8.4.82` detects sensible objects on the canonical
photo and standard images, and detects nothing on the blank image.

## Preprocess Parity

Explicit OpenCV square letterbox preprocessing matches Ultralytics
`LetterBox(auto=False, center=True)` byte-for-byte on the oracle suite. This
removes preprocessing mismatch as the cause of the earlier false result.

## Canonical False Refrigerator Root Cause

The false `refrigerator` is reproduced by `8.3.233` but not by `8.4.0` or
`8.4.82`. The latest traditional raw class-score probe shows high `laptop` and
`person` scores and near-zero `refrigerator` score, while `8.3.233` has
`refrigerator` as its strongest class.

Conclusion: version mismatch caused wrong behavior and the newer package path
fixes the CPU oracle.

## INT8 CPU Oracle Update

The latest INT8 pass found two different outcomes:

- Ultralytics `quantize=8` is not an acceptable YOLO26 INT8 oracle path yet.
  It exported Q/DQ ONNX models for both output contracts, but CPU ORT decoded
  zero detections on all nonblank oracle images.
- Manual ONNX Runtime static Q/DQ over `Conv`/`MatMul` produced CPU-good INT8
  candidates with meaningful detections on canonical, bus, zidane, and camera
  still inputs, while keeping the blank image clean for the accepted candidates.

The CPU-good manual Q/DQ models are evidence that INT8 is not fundamentally
blocked at the model/decode layer. The current blocker is rt204 SpaceMIT EP
compilation of those Q/DQ subgraphs, documented in `docs/RUNTIME_204_NOTES.md`.
