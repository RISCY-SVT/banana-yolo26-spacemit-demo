# YOLO26 INT8 Status

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
/data/ncnn-logs/ort-logs/2026-06-30_07-56-51/
/data/ncnn-logs/ort-logs/2026-06-30_08-45-33/
/data/ncnn-logs/ort-logs/2026-06-30_09-38-36/
/data/ncnn-logs/ort-logs/2026-06-30_14-21-27/
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

## XSlim Gate Addendum

SpacemiT XSlim `2.1.0` was evaluated after the INT8 closure decision.

- Static XSlim PTQ did not produce an accepted YOLO26 INT8 model in this gate.
  The e2e `[1,300,6]` export fails inside XSlim/PPQ `ReduceMax`; the
  traditional `[1,84,8400]` path entered long block-wise calibration without a
  practical bounded output.
- XSlim `--dynq` produced CPU-good and rt204-runnable e2e/traditional models.
  These are diagnostic dynamic/weight-dequantized graphs with normal `Conv`
  nodes and `DequantizeLinear` weights, not static Q/DQ or QOperator INT8
  graphs.
- Because no CPU-good static XSlim INT8 model reached rt204 EP, XSlim does not
  change the current decision.

Current XSlim decision:

```text
XSLIM_YOLO26_INT8_PARTIAL_FALLBACK_ONLY
```

## XSlim Static PTQ Follow-Up

The follow-up XSlim static PTQ pass refined the decision:

```text
XSLIM_STATIC_YOLO26_INT8_NEEDS_UPSTREAM_FIX
```

- The e2e `[1,300,6]` path fails in XSlim/PPQ `ReduceMax` handling on both
  XSlim `2.1.0` and main/`2.1.1`. The failure reproduces with tiny ONNX models
  after XSlim converts ReduceMax to a two-input form.
- Config-level workarounds (`ignore_op_types`, `ignore_op_names`,
  `skip_onnxsim`, `opset=18`, `calibration_type=minmax`) do not avoid the e2e
  failure.
- Traditional `[1,84,8400]` static PTQ with `calibration_type=minmax` and
  `percentile` now emits Q/DQ ONNX models, but both are CPU-bad: class score
  channels are all zero on public sanity images even at very low confidence.
- A diagnostic rt204 run of the CPU-bad minmax model executes, proving that this
  XSlim graph shape can avoid the previous `clip minmax` compile error, but it
  is not a usable INT8 path because CPU semantics already failed.

No CPU-good and rt204-EP-good XSlim static YOLO26 INT8 candidate exists.

## FP32/FP16 Effect Matrix Addendum

The FP32/FP16/XSlim effect pass did not reopen INT8 implementation work. It
confirmed the closure status only:

- INT8 acceleration remains blocked pending upstream/vendor fixes.
- The vendor Q/DQ Conv `clip minmax` repro remains the actionable rt204 issue.
- XSlim static PTQ remains blocked by e2e `ReduceMax` handling and
  traditional zero-score CPU oracle.
- XSlim dynamic quantization remains diagnostic only.
- No INT8 FPS is claimed.
