# ONNX LUT Reference Oracle Report

classification: pass

Scope: host-side 256-code Q/DQ+SiLU LUT oracle only. This does not add ONNX Runtime to the C++ runtime library.

Tooling:

- venv: `.deps/custom_int8_engine/venv-stage7-onnx`
- `onnx=1.22.0`
- `onnxruntime=1.27.0`
- `numpy=2.5.0`

Standalone ONNX graphs were generated under the Stage 9 log dir:

| boundary | ONNX path | SHA256 | mismatches vs Stage 8 LUT | max_abs_diff |
|---|---|---|---:|---:|
| Act0 | `run_logs/act0_qdq_silu_lut.onnx` | `5ef2c202ff4050bee94730dc2fb0665d393401b0b7270b4e3ab05b35224b1446` | 0 | 0 |
| Act1 | `run_logs/act1_qdq_silu_lut.onnx` | `a5882f37fa7630ec2f3fbc6ea76cfd6010f5beb44e459a98a37c60903bf4c905` | 0 | 0 |

Boundary graph:

```text
uint8 conv_output_code
DequantizeLinear
Sigmoid
Mul
QuantizeLinear
uint8 activation_code
signed storage = activation_code - 128
```

Result: ONNX Runtime CPU matched the Stage 8 internal scalar LUT reference for all 256 input codes on Act0 and Act1.
