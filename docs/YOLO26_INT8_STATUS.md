# YOLO26 INT8 Status

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
```

## Current Decision

```text
YOLO26 INT8 ONNX board acceleration is blocked pending rt204/vendor EP compiler support.
```

## What Works

- YOLO26 FP32 PyTorch oracle is good with current Ultralytics.
- YOLO26 FP32 ONNX Runtime CPU oracle is good.
- YOLO26 FP32 rt204 SpaceMIT EP executes on BPI-F3/K1X without SIGILL and
  matches CPU-level semantics.
- Manual ONNX Runtime static Q/DQ can produce CPU-good INT8 candidates, so the
  model is not fundamentally unquantizable.

## What Fails

- Ultralytics `quantize=8` candidates produced Q/DQ ONNX models whose CPU
  oracle collapsed to zero detections in the tested calibration attempts.
- CPU-good manual Q/DQ INT8 models fail under rt204 SpaceMIT EP with:

```text
output_type not implemented for clip minmax
```

- QOperator candidates are not performance-gate ready: useful `QLinearConv` /
  `QLinearMatMul` offload was not visible, CPU/EP parity was loose, and bounded
  smoke was slower than FP32.

## Minimal Repros

| Repro | Role | Result |
| --- | --- | --- |
| `15_conv_qdq_attr_kernel_shape.onnx` | Tiny synthetic Q/DQ Conv repro | CPU ORT passes; rt204 EP fails with `clip minmax`. |
| `07_yolo26_first_conv_qdq_output_block.onnx` | Smallest real YOLO26-derived repro | CPU ORT passes; rt204 EP fails at `/model.0/conv/Conv_token_1`. |

The isolated trigger is Q/DQ Conv with explicit `kernel_shape=[3,3]`.

## Fallbacks

| Fallback | Status |
| --- | --- |
| `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv` | Correctness-only fallback; CPU-heavy; not accelerated INT8. |
| Strip optional Conv `kernel_shape` + `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add` | Partial diagnostic candidate; avoids first Conv blocker but still unproven for useful placement/perf. |
| QOperator | Rejected for current perf gate. |

## Decision

YOLO26 INT8 should not move to board performance benchmarking as an accelerated
INT8 path until rt204/vendor EP compiler support changes or the partial
fallback is proven in a separate placement/performance gate.
