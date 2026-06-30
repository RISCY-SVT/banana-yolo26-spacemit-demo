# RT204 Q/DQ Conv Vendor Bug Report

Vendor bundle:

```text
/data/ncnn-logs/ort-logs/2026-06-30_06-12-26/artifacts/vendor_bug_report_rt204_qdq_conv_clip_minmax/
```

## Environment

| Item | Value |
| --- | --- |
| Board | Banana-Pi BPI-F3 |
| SoC | SpacemiT K1X / X60 |
| Runtime | `spacemit-ort.riscv64.2.0.4` |
| Provider | SpaceMITExecutionProvider |
| Failure class | Q/DQ Conv compile failure |

## Error

```text
output_type not implemented for clip minmax
```

## Bundle Contents

The bundle uses subdirectories for models and inputs:

```text
models/15_conv_qdq_attr_kernel_shape.onnx
models/07_yolo26_first_conv_qdq_output_block.onnx
inputs/input_1x3x8x8_f32.bin
inputs/canonical_640_nchw_f32.bin
expected_cpu_output.txt
actual_rt204_error.txt
board_info.txt
runtime_info.txt
provider_options_inventory.txt
sha256sums.txt
run_cpu.sh
run_rt204.sh
README.md
```

## Minimal Repro

The smallest synthetic repro is:

```text
models/15_conv_qdq_attr_kernel_shape.onnx
```

It is a tiny Q/DQ Conv graph with an explicit Conv `kernel_shape=[3,3]`
attribute. CPU ORT executes it correctly. rt204 SpaceMIT EP fails the compile
path with the `clip minmax` error above.

The supplemental real-model repro is:

```text
models/07_yolo26_first_conv_qdq_output_block.onnx
```

It is extracted from YOLO26 first Conv and fails at:

```text
/model.0/conv/Conv_token_1
```

## Why This Blocks YOLO26 INT8

Manual ONNX Runtime static Q/DQ quantization can produce CPU-good YOLO26 INT8
models, but the rt204 EP selects the Q/DQ Conv subgraph and fails during
provider compilation before useful board INT8 execution. This is a runtime/EP
compiler support blocker, not a YOLO26 decode or preprocessing issue.

## Known Correctness-Only Workaround

```bash
SPACEMIT_EP_DISABLE_OP_TYPE_FILTER=QuantizeLinear;DequantizeLinear;Conv
```

This restores correctness by moving the problematic Q/DQ Conv regions away
from SpaceMIT EP. It is CPU-heavy and must not be described as accelerated
YOLO26 INT8.

## XSlim Static PTQ Addendum

The XSlim static PTQ follow-up did not close this vendor blocker.

XSlim traditional `minmax` static PTQ generated a Q/DQ ONNX model that runs on
rt204 without the `clip minmax` compile error, but the model is CPU-bad: class
score channels are all zero before rt204 is involved. This is useful as a
separate XSlim calibration/export finding, but it does not invalidate the
vendor-ready rt204 repros for CPU-good manual ONNX Runtime Q/DQ models.

The original rt204 vendor repro bundle remains the evidence to send for the
runtime/compiler issue.
