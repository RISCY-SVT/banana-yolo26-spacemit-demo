# RT204 Operator Support Notes

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
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

The 2026-06-29 minimization pass narrowed this further:

- toy Conv/Q/DQ/Clip/QLinearConv models pass on rt204;
- extracted real YOLO26 first-block Q/DQ Conv fails at
  `/model.0/conv/Conv_token_1`;
- filtering only the first Conv in a two-Conv extracted block moves the failure
  to the next Conv;
- pass filters for `AddQDQPreMinmaxParams` and `QLinearLeaglization` did not
  avoid the compile failure.

The 2026-06-30 vendor-repro pass reduced the blocker to a tiny synthetic Q/DQ
Conv model with explicit `kernel_shape=[3,3]`:

```text
15_conv_qdq_attr_kernel_shape.onnx
```

Attr isolation showed that `kernel_shape` is sufficient to trigger the
`clip minmax` failure. `dilations` alone and `group` alone pass.

## YOLO26 INT8 Fallbacks

The smallest correct Q/DQ fallback found so far is:

```bash
SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv
```

This keeps semantics correct by pushing the problematic Q/DQ Conv regions away
from SpaceMIT EP. It is CPU-heavy and should not be treated as an accelerated
INT8 path.

An end-to-end QOperator Conv+MatMul candidate runs under rt204, but the gate
rejected it for performance benchmarking: raw CPU/EP parity is loose, the bus
oracle changes top semantics under EP, dumped subgraphs do not contain
`QLinearConv` or `QLinearMatMul`, and bounded timing smoke is slower than FP32.

Removing optional Conv `kernel_shape` attributes from the full Q/DQ model is
CPU-exact and avoids the first Conv blocker, but exposes an attention MatMul
runtime issue:

```text
cannot find kernel config for this vlen 256 and weight type u8
```

`SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add` restores smoke semantics for
that stripped model. This is a partial fallback candidate only, not a proven
accelerated INT8 path.

## YOLO11 RT204 Signal

Frozen YOLO11 production models were probed read-only in this R&D workspace:

- dynamic640 INT8;
- vendor320 q.onnx;
- FP16 keep_io 640.

The direct rt204 tensor probe produced sane CPU and SpaceMIT EP semantics for
all three. This is promising, but it is not enough to change production policy:
a future YOLO11 rt204 adoption gate must rerun the full production app, loader,
camera, and performance matrix.
