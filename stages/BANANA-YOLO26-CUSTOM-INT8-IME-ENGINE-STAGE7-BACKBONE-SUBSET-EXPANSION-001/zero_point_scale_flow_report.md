# Zero Point And Scale Flow Report

Input model: `/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`

Recovered from Stage 6 and Stage 7 graph inspection:

| Boundary | Scale | Zero point | Storage for IME | Notes |
|---|---:|---:|---:|---|
| `images` | `0.00392156885937` | `0` | `-128` signed int8 | model input Q/DQ |
| `/model.0/conv` output | `0.620968401432` | `128` | activation fallback input | per-tensor output Q |
| `/model.0/act/Mul` output | `0.311162889004` | `1` | `-127` signed int8 | feeds `/model.1/conv` |
| `/model.1/conv` output | `1.1142437458` | `122` | activation fallback input | per-tensor output Q |
| `/model.1/act/Mul` output | `0.582500994205` | `0` | `-128` signed int8 | feeds `/model.2/cv1/conv` |
| `/model.2/cv1/conv` output | `0.836495816708` | `155` | output boundary deferred | Stage 7 compares corrected int32/dequant |

Weights for selected Conv nodes use signed int8 with zero-point 0 per Stage 6/7 extraction. Raw IME kernels compute `sum(a_s8 * w_s8)`; zero-point correction is applied via the existing selected-node correction path before activation/requant or output comparison.

No full-graph requant policy is claimed in Stage 7.
