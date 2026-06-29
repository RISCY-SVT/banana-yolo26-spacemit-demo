# RT204 Operator Support Notes

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
```

## YOLO26 Float Coverage

RT204 SpaceMIT EP compiles and executes the latest YOLO26 640 FP32 exports:

- end-to-end `[1,300,6]`;
- traditional `[1,84,8400]`.

The float subgraphs include the main YOLO26 Conv/Sigmoid/Mul head and decode
regions, and decoded semantics match CPU-level expectations on canonical, bus,
and blank images.

## YOLO26 INT8 Coverage

Manual CPU-good Q/DQ INT8 subgraphs are selected by SpaceMIT EP, but compilation
fails before execution:

```text
output_type not implemented for clip minmax
```

Observed Q/DQ-heavy subgraphs include approximately:

- end-to-end Q/DQ: 1080 nodes, 454 `DequantizeLinear`, 250 `QuantizeLinear`,
  98 `Conv`, 4 `MatMul`;
- traditional Q/DQ: 1079 nodes, 454 `DequantizeLinear`, 250 `QuantizeLinear`,
  98 `Conv`, 4 `MatMul`.

This means the current blocker is not decode, preprocessing, or CPU INT8
correctness. It is rt204 EP support for the quantized Conv/Q/DQ pattern emitted
by manual ONNX Runtime static quantization.

## YOLO11 RT204 Signal

Frozen YOLO11 production models were probed read-only in this R&D workspace:

- dynamic640 INT8;
- vendor320 q.onnx;
- FP16 keep_io 640.

The direct rt204 tensor probe produced sane CPU and SpaceMIT EP semantics for
all three. This is promising, but it is not enough to change production policy:
a future YOLO11 rt204 adoption gate must rerun the full production app, loader,
camera, and performance matrix.
