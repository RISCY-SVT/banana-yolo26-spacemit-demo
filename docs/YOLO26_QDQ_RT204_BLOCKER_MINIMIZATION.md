# YOLO26 Q/DQ RT204 Blocker Minimization

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
```

## Scope

This R&D pass minimized the rt204 SpaceMIT EP failure seen with CPU-good
YOLO26 INT8 Q/DQ models. It did not modify the frozen YOLO11 production repo,
does not change production policy, and does not claim YOLO26 production
readiness.

## Starting Point

The previous pass established:

- latest YOLO26 FP32 CPU oracle is correct;
- rt204 FP32 SpaceMIT EP matches CPU-level semantics for YOLO26 640/320;
- Ultralytics `quantize=8` Q/DQ exports are CPU-bad, producing zero
  detections;
- manual ONNX Runtime static Q/DQ can produce CPU-good YOLO26 INT8 candidates;
- rt204 SpaceMIT EP fails to compile those CPU-good Q/DQ models with
  `output_type not implemented for clip minmax`.

## Minimal Reproduction Result

Synthetic minimal ONNX models were not enough to reproduce the failure:

- single Conv FP32;
- single Conv with Q/DQ;
- QLinearConv wrapper;
- Conv + Clip;
- Conv + Q/DQ + Clip.

All of those passed under rt204 SpaceMIT EP. The minimal repro had to be
extracted from the real YOLO26 Q/DQ graph:

```text
yolo26_first_conv_qdq_output_block.onnx
```

That model contains only the first YOLO26 quantized Conv block and surrounding
Q/DQ nodes. It fails under rt204 SpaceMIT EP at:

```text
/model.0/conv/Conv_token_1
output_type not implemented for clip minmax
```

The two-Conv extracted block shows the same behavior: filtering the first Conv
only moves the failure to the next quantized Conv. This proves the blocker is
not a single bad node name; it is the rt204 compiler path for the real YOLO26
Q/DQ Conv pattern.

## Export and Quantization Settings

Additional bounded quantization attempts did not avoid the blocker:

- Q/DQ Conv+MatMul per-tensor;
- Q/DQ Conv+MatMul reduce-range;
- Q/DQ with early Conv nodes excluded;
- traditional Q/DQ Conv-only;
- MatMul-only Q/DQ where applicable.

Excluding the first Conv or the first eight Conv nodes only moved the failure
to the next quantized Conv. Provider pass filters such as
`AddQDQPreMinmaxParams` and `QLinearLeaglization` did not fix compilation.

## Provider Filter Findings

Full-model filter results:

- default rt204: Q/DQ Conv models fail with `output_type not implemented for
  clip minmax`;
- `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=Conv`: not enough for Conv+MatMul Q/DQ;
- `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear`:
  executes but overproduces invalid detections;
- `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv`:
  restores semantic correctness by forcing CPU-heavy fallback.

The smallest correct Q/DQ fallback is therefore:

```bash
SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv
```

This is useful for diagnosis and correctness fallback only. It is not an
accelerated board INT8 path.

## QOperator Fallback Signal

An end-to-end QOperator candidate over Conv+MatMul runs under rt204 and gives
sane semantics on canonical, bus, and blank oracle images. However:

- raw CPU/EP output parity is loose;
- dumped EP subgraphs do not show clear `QLinearConv` or `QLinearMatMul`
  offload;
- perf smoke on generated input was slower than FP32.

This path is a partial fallback candidate, not a replacement for a correct
accelerated Q/DQ INT8 path. It should enter a separate focused fallback gate
before any performance claims.

## Decision

```text
YOLO26 INT8 status: partial
```

Manual INT8 CPU oracle is good, but rt204 Q/DQ Conv full-offload is blocked.
YOLO26 INT8 is not ready for board performance benchmarking as an accelerated
Q/DQ path.

The next useful task is:

```text
BANANA-YOLO26-RT204-QDQ-CONV-VENDOR-REPRO-AND-QOPERATOR-GATE-001
```
