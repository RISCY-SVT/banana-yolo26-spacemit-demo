# YOLO26 INT8 and RT204 Forensics

Raw evidence:

```text
/data/ncnn-logs/ort-logs/2026-06-29_16-56-34/
/data/ncnn-logs/ort-logs/2026-06-29_21-43-36/
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/
```

## Scope

This R&D pass investigated YOLO26n INT8 export/calibration and SpacemiT ORT
`2.0.4` operator support. It did not modify the frozen YOLO11 production repo
and it does not claim YOLO26 production readiness.

## Ultralytics INT8 Result

Selected package:

```text
ultralytics==8.4.82
yolo26n.pt sha256=9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

Tested Ultralytics `quantize=8` exports covered:

- end-to-end `[1,300,6]`;
- traditional `[1,84,8400]`;
- task-local `calib_small` and `calib_representative` YAML files;
- `fraction=1.0` and a representative `fraction=0.25` smoke;
- `simplify=True/False`;
- `nms=True` where accepted.

All Ultralytics Q/DQ candidates produced zero detections in CPU ORT on nonblank
oracle images. Score tensors collapsed to zero while coordinate ranges remained
nonzero. This path is not usable as a YOLO26 INT8 oracle.

## Manual ORT Q/DQ Result

Manual ONNX Runtime static Q/DQ over `Conv` and `MatMul` recovered the CPU
oracle:

- `manual_e2e_rep_conv_matmul_qdq`: end-to-end `[1,300,6]`, representative
  calibration, CPU-good, blank-clean;
- `manual_e2e_small_conv_matmul_qdq`: end-to-end `[1,300,6]`, small
  calibration, CPU-good, blank-clean;
- `manual_trad_rep_conv_matmul_qdq`: traditional `[1,84,8400]`,
  representative calibration, CPU-good, blank-clean;
- `manual_trad_rep_conv_only_qdq`: traditional `[1,84,8400]`, representative
  calibration, CPU-good, blank-clean.

`manual_e2e_rep_conv_only_qdq` was rejected because it produced a blank-image
false positive.

## RT204 Board EP Result

Float YOLO26 640 end-to-end and traditional ONNX models run on rt204 SpaceMIT
EP with semantic parity against CPU.

CPU-good manual Q/DQ INT8 candidates fail under rt204 SpaceMIT EP with:

```text
output_type not implemented for clip minmax
```

The failure occurs while executing the EP subgraph at the first Conv token. The
dumped subgraphs contain Q/DQ-heavy Conv/MatMul regions; the dumped ONNX graph
does not expose a user-authored `Clip` op at that point, so the message appears
to describe an internal compiler min/max handling path for quantized Conv/QDQ.

Provider filter diagnostics:

- disabling only `QuantizeLinear;DequantizeLinear` allows execution but
  overproduces invalid detections;
- disabling only `Conv` still fails compilation;
- disabling `QuantizeLinear;DequantizeLinear;Conv` recovers CPU-like semantics
  by forcing CPU-heavy fallback, but it is slower and not an accelerated INT8
  solution.

## Decision

YOLO26 INT8 board EP is blocked by rt204 EP compilation of CPU-good Q/DQ INT8
subgraphs. The next useful work is a quantization-pattern reduction pass or a
vendor-supported SpacemiT ORT 2.0.4 YOLO26 INT8 recipe.

## 2026-06-29 Q/DQ Blocker Minimization Update

The follow-up minimization pass reduced the blocker to an extracted real YOLO26
first Conv Q/DQ block:

```text
yolo26_first_conv_qdq_output_block.onnx
/model.0/conv/Conv_token_1
output_type not implemented for clip minmax
```

The initial synthetic toy Conv/Q/DQ/Clip/QLinearConv set passed on rt204, so
the failure was not reproduced by a generic no-attribute toy graph. The
2026-06-30 attr-isolation pass later found that explicit Conv
`kernel_shape=[3,3]` is sufficient to reproduce the same rt204 internal min/max
handling failure.

Bounded export and quantization changes did not remove the blocker. The
smallest correct Q/DQ fallback is:

```bash
SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv
```

That fallback restores semantics through CPU-heavy execution and is not an
accelerated INT8 benchmark path.

See `docs/YOLO26_QDQ_RT204_BLOCKER_MINIMIZATION.md` for the detailed decision.

## 2026-06-30 Vendor-Repro and QOperator Gate Update

The smallest current reproducer is no longer the real YOLO26 first Conv block.
The failure is reproduced by a tiny synthetic Q/DQ Conv model with explicit
`kernel_shape=[3,3]`:

```text
15_conv_qdq_attr_kernel_shape.onnx
```

The real YOLO26 first Conv block remains as a supplemental repro. The exact
failure is still:

```text
output_type not implemented for clip minmax
```

QOperator was gated and rejected for the next performance stage because useful
`QLinearConv`/`QLinearMatMul` offload was not visible in dumped subgraphs, raw
CPU/EP parity was loose, and the bus oracle changed top semantics under EP.

Stripping optional Conv `kernel_shape` attributes from the full Q/DQ model is
CPU-exact and avoids the first Conv blocker, but a second rt204 issue appears
in the attention MatMul path. A partial fallback with
`SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=MatMul;Add` restores smoke semantics for
that stripped model, but it remains a future placement/performance-gate
candidate rather than a ready INT8 benchmark path.
