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

RT204 INT8 is still not claimed.

The 2026-06-29 INT8/operator-support pass separated three layers:

- Ultralytics `quantize=8` exports Q/DQ ONNX files, but every tested candidate
  collapses confidence/class scores to zero in the CPU oracle.
- Manual ONNX Runtime static Q/DQ over `Conv` and `MatMul` produces CPU-good
  YOLO26 INT8 candidates for both end-to-end `[1,300,6]` and traditional
  `[1,84,8400]` contracts.
- RT204 SpaceMIT EP fails to compile those CPU-good Q/DQ candidates on board
  with `output_type not implemented for clip minmax` at the first Conv token in
  the EP subgraph.

Provider filters confirm the failure is in the EP-compiled Q/DQ/Conv subgraph:
disabling Q/DQ alone runs but overproduces invalid detections; disabling Q/DQ
and Conv together recovers CPU-like correctness through fallback, but that path
is slower and is not an accelerated INT8 solution.

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
```

## YOLO11 Reevaluation Signal

Frozen YOLO11 production models were copied read-only into the YOLO26 R&D board
stage and run through a direct rt204 tensor probe. Dynamic640 INT8, vendor320
q.onnx, and FP16 keep_io 640 all returned sane CPU and SpaceMIT EP semantics.

This does not change YOLO11 production policy. It is a future rt204 adoption
gate candidate that still needs the full production app, loader, camera, and
performance matrix.
