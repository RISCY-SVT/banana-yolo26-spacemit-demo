# Stage 15 Boundary LUT Oracle Report

provider: `CPUExecutionProvider`

| boundary | micro_model | micro_model_sha256 | mismatches | max_abs_diff_u8 |
|---|---|---:|---:|---:|
| `/model.4/cv1/conv` -> `/model.4/Split_output_1` | `.deps/custom_int8_engine/stage15_oracle/stage15_model4_cv1_to_split1_activation.onnx` | `15aeef7bcd46f8a77ce98091171a153ae26f2e69fab650f0de69c02125924583` | `0` | `0` |
| `/model.4/m.0/cv1/conv` -> `/model.4/m.0/cv1/act` | `.deps/custom_int8_engine/stage15_oracle/stage15_model4_m0_cv1_activation.onnx` | `9549703601a63b32d2dc57f519594ed55d05b0a3fe24c8a8d08ef50c942ae7a1` | `0` | `0` |

Both accepted boundaries require `mismatches=0` and `max_abs_diff_u8=0`.
