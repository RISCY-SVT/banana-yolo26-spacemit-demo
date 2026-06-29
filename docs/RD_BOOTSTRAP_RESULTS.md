# YOLO26 R&D Bootstrap Results

Task: `BANANA-YOLO26-SPACEMIT-DEMO-RD-BOOTSTRAP-001`

Raw evidence directory:

```text
/data/ncnn-logs/ort-logs/2026-06-29_15-01-19/
```

## Summary

- The frozen YOLO11 production repository was verified read-only at commit
  `9c0933be58ee122389d1a43f45f81e80655d6904` and tag
  `production-2026-07-02`.
- This repository was created from the YOLO11 production tag on branch
  `yolo26-rd-bootstrap`; the inherited remote is renamed to
  `template-yolo11-gitlab`.
- Public SpacemiT ORT `2.0.4` was downloaded and inventoried.
- RT204 adds `SPACEMIT_EP_DISABLE_PASSES_FILTER` and new strings/symbols for
  `YoloDecode`, `GridSample`, `RotaryEmbedding`, and `ArgMax`.
- RT204 runs on BPI-F3/K1X/X60 with default architecture selection; tested text
  and numeric `SPACEMIT_EP_PERFER_CORE_ARCH` overrides were rejected.
- `yolo26n.pt` was downloaded from the Ultralytics public assets release
  `v8.4.0` and exported to ONNX at 640 and 320.
- The installed Ultralytics exporter produced traditional outputs
  `[1,84,8400]` and `[1,84,2100]`; `end2end=False` was not accepted as an
  export argument in the tested package.
- CPU decode matches Ultralytics for the exported 640 graph, but both produce a
  giant false `refrigerator` on the canonical photo.
- RT204 SpaceMIT EP runs the exported YOLO26 640/320 graphs without crashing and
  reproduces CPU-level semantics, including the bad visual result.
- INT8 was intentionally not attempted because the float CPU oracle is already
  semantically bad.

## Next Gate

Before quantization or production-style demo work, obtain or produce a YOLO26
export whose CPU oracle is visually sane on the canonical photo and resolve the
documented `(N,300,6)` end-to-end contract mismatch.
