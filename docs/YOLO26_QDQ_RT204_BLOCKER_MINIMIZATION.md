# YOLO26 Q/DQ RT204 Blocker Minimization

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
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

The 2026-06-29 pass first proved that basic synthetic ONNX models were not
enough to reproduce the failure:

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

The 2026-06-30 vendor-repro pass reduced the trigger further to a tiny
synthetic model:

```text
15_conv_qdq_attr_kernel_shape.onnx
```

That model is a small Q/DQ Conv graph with an explicit Conv
`kernel_shape=[3,3]` attribute. CPU ORT runs it, while rt204 SpaceMIT EP fails
with the same:

```text
output_type not implemented for clip minmax
```

The attr-isolation matrix showed that `kernel_shape` alone is sufficient to
trigger the failure; `dilations` alone and `group` alone pass.

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

The follow-up QOperator gate showed that this path is not ready for a
performance gate. The dumped EP subgraphs do not include `QLinearConv` or
`QLinearMatMul`, CPU/EP raw parity is loose, the bus oracle changes top
semantics under EP, and bounded timing smoke is slower than FP32.

## Strip-Kernel Partial Fallback Signal

Removing optional Conv `kernel_shape` attributes from the full YOLO26 Q/DQ
model is CPU-exact relative to the original Q/DQ model. It avoids the first
Conv `clip minmax` compiler failure, but exposes a second rt204 issue:

```text
/model.10/m/m.0/attn/MatMul
cannot find kernel config for this vlen 256 and weight type u8
```

The smallest correct fallback seen for that stripped model is:

```bash
SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add
```

This is a partial fallback candidate only. Bounded timing smoke is slower than
FP32, and placement/performance must be proven in a separate gate before any
benchmarking claim.

## Decision

```text
YOLO26 INT8 status: partial
```

Manual INT8 CPU oracle is good, but rt204 Q/DQ full-offload is blocked. YOLO26
INT8 is not ready for full board performance benchmarking as an accelerated
Q/DQ path.

The next useful task is:

```text
BANANA-YOLO26-RT204-PARTIAL-FALLBACK-PERF-PLACEMENT-GATE-001
```
