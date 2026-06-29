# SpacemiT ORT 2.0.4 Notes

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-03-40/
```

## Runtime Status

RT204 was already bootstrapped from the official GitHub release asset:

```text
spacemit-ort.riscv64.2.0.4.tar.gz
sha256=bcf02bd12b8a1df969d6986658a9270c1121e5d58f5947d91ea5eba1bd6cd435
```

The runtime adds or exposes strings/options including:

- `SPACEMIT_EP_DISABLE_PASSES_FILTER`;
- `YoloDecode`;
- `GridSample`;
- `RotaryEmbedding`;
- `ArgMax`.

## K1X / X60 Behavior

RT204 runs on BPI-F3/K1X/X60 in default mode without SIGILL for the tested
session probe and YOLO26 FP32 exports. The discovered
`SPACEMIT_EP_PERFER_CORE_ARCH` option exists, but useful public K1/K3/X60
override values were not found in the previous bootstrap.

## YOLO26 FP32 Parity

Latest YOLO26 640 end-to-end and traditional ONNX exports run with both CPU and
SpaceMIT EP providers. On canonical, bus, and blank inputs the decoded semantic
results match CPU-level expectations:

- end-to-end canonical: top `laptop`, 11 detections;
- traditional canonical: top `laptop`, 12 detections;
- bus: top `bus`;
- blank: zero detections.

## INT8 Status

RT204 INT8 is not claimed. Ultralytics `quantize=8` produced a Q/DQ ONNX, but
host ORT CPU returned zero detections on the oracle suite and the default export
attempted to auto-download `coco8`. A task-local calibration path is required
before board INT8 testing.
