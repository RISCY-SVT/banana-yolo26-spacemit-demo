# Tensor Contract Report

Selected subset: `candidate_C_block0_silu_model1_conv`

Runtime layout for custom runner: NHWC int8 storage for activation tensors, OHWI int8 storage for weights, NHWC int32 for Conv accumulator outputs.

| tensor_name | graph shape | graph dtype | producer | consumer | scale | zero_point | signedness/storage | lifetime | workspace_needed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `images` | `[1,3,640,640]` | float | model input | `images_QuantizeLinear` | n/a | n/a | float NCHW in ONNX tooling only | input | no runtime dependency |
| `images_QuantizeLinear_Output` | `[1,3,640,640]` | uint8 | `images_QuantizeLinear` | `images_DequantizeLinear` | `images_scale=0.00392156885937` | `images_zero_point=0` | stored as int8 `q_u8 - 128`, zp_s8 `-128` | hot input | caller-provided |
| `/model.0/conv/Conv_output_0` | `[1,16,320,320]` | float | `/model.0/conv/Conv` | Q/DQ | accumulator scale per OC: `images_scale * model.0.conv.weight_scale[oc]` | raw int32 correction uses activation zp `0` and weight zp `0` | corrected int32 then float for Q/DQ | hot intermediate | `conv0_i32` |
| `/model.0/conv/Conv_output_0_QuantizeLinear_Output` | `[1,16,320,320]` | uint8 | Conv0 Q | Conv0 DQ | `0.620968401432` | `128` | uint8 quantized Conv0 output | hot intermediate | `conv0_q_s8` can be fused through activation |
| `/model.0/conv/Conv_output_0_DequantizeLinear_Output` | `[1,16,320,320]` | float | Conv0 DQ | `Sigmoid`, `Mul` | `0.620968401432` | `128` | float fallback in Stage 6 activation path | hot intermediate | no persistent full float buffer required in final path |
| `/model.0/act/Sigmoid_output_0` | `[1,16,320,320]` | float | `/model.0/act/Sigmoid` | `/model.0/act/Mul` | float | n/a | float fallback | transient | fused in activation loop |
| `/model.0/act/Mul_output_0` | `[1,16,320,320]` | float | `/model.0/act/Mul` | Act Q/DQ | float | n/a | float fallback | transient | fused in activation loop |
| `/model.0/act/Mul_output_0_QuantizeLinear_Output` | `[1,16,320,320]` | uint8 | Act Q | Act DQ | `0.311162889004` | `1` | stored as int8 `q_u8 - 128`, zp_s8 `-127` | hot Conv1 input | `conv1_input_s8` |
| `/model.0/act/Mul_output_0_DequantizeLinear_Output` | `[1,16,320,320]` | float | Act DQ | `/model.1/conv/Conv` | `0.311162889004` | `1` | Conv1 quantized activation represented as int8 storage | hot Conv1 input | `conv1_input_s8` |
| `/model.1/conv/Conv_output_0` | `[1,32,160,160]` | float | `/model.1/conv/Conv` | Conv1 Q/DQ | accumulator scale per OC: `act0_scale * model.1.conv.weight_scale[oc]` | raw int32 correction uses activation zp `1` and weight zp `0` | corrected int32 output boundary | output | `conv1_i32` |

Per-channel weights:

- `model.0.conv.weight_quantized`: shape `[16,3,3,3]`, per-output-channel int8, zero-point min/max `0/0`.
- `model.1.conv.weight_quantized`: shape `[32,16,3,3]`, per-output-channel int8, zero-point min/max `0/0`.

Large tensor dumps are not committed. Stage 6 fixture headers contain only small deterministic ROI fixtures.

