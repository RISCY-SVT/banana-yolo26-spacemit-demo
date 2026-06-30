# RT204 Q/DQ Conv Vendor Repro and QOperator Gate

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
```

## Scope

This stage turned the YOLO26 rt204 Q/DQ failure into a vendor-grade repro and
gated the two plausible escape routes:

- QOperator quantization;
- selective provider fallback after graph rewrites.

It did not modify the frozen YOLO11 production repository and it does not make
YOLO26 production claims.

## Minimal Repro Result

A tiny synthetic repro now exists:

```text
15_conv_qdq_attr_kernel_shape.onnx
```

The model is a small Q/DQ Conv graph with an explicit Conv
`kernel_shape=[3,3]` attribute. CPU ORT executes it correctly, but rt204
SpaceMIT EP fails the Conv compile path with:

```text
output_type not implemented for clip minmax
```

The attr-isolation matrix showed:

- `kernel_shape` alone triggers the failure;
- `dilations` alone passes;
- `group` alone passes;
- `kernel_shape+dilations` fails;
- `kernel_shape+group` fails;
- `dilations+group` passes.

The supplemental real-graph repro remains:

```text
yolo26_first_conv_qdq_output_block.onnx
```

That model is the extracted first YOLO26 Q/DQ Conv block and fails at:

```text
/model.0/conv/Conv_token_1
output_type not implemented for clip minmax
```

## QOperator Gate

The QOperator e2e and traditional candidates are not ready for a performance
gate:

- CPU ORT runs them and gives plausible detections;
- rt204 EP runs them, but raw CPU/EP outputs do not match;
- the bus oracle image changes top semantics under EP;
- dumped SpaceMIT subgraphs contain no `QLinearConv` or `QLinearMatMul`;
- bounded timing smoke is slower than the current FP32 e2e probe.

Decision:

```text
QOPERATOR_CPU_ONLY_NOT_ACCELERATED
```

## Partial Fallback Findings

The original CPU-heavy fallback remains correct but not useful for accelerated
INT8:

```bash
SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv
```

Removing optional `kernel_shape` attributes from the full YOLO26 Q/DQ model is
CPU-exact relative to the original Q/DQ model. It avoids the first Conv
`clip minmax` failure, but exposes a second rt204 issue in the attention
MatMul path:

```text
/model.10/m/m.0/attn/MatMul
cannot find kernel config for this vlen 256 and weight type u8
```

The smallest correct fallback after stripping `kernel_shape` is currently:

```bash
SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add
```

This path restores semantic output on the canonical smoke image, but the
bounded timing smoke is slower than FP32 and it still needs a separate
placement/performance gate before it can be treated as useful acceleration.

## Decision

YOLO26 INT8 is not ready for full board performance benchmarking.

The current status is:

```text
YOLO26 INT8 partial fallback candidate only: yes
```

The next useful task is a narrow partial-fallback placement/performance gate for
the stripped-kernel model, or vendor feedback on the tiny `kernel_shape` Q/DQ
Conv repro.
