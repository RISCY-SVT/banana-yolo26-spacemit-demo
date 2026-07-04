# Concat Contract Report

node_name: `/model.2/Concat`
op_type: `Concat`
axis: `1` in ONNX NCHW

## Inputs

| order | tensor | shape | domain |
|---:|---|---:|---|
| 0 | `/model.2/Split_output_0` | `[1,16,160,160]` | float direct from Split, no Q/DQ |
| 1 | `/model.2/Split_output_1_DequantizeLinear_Output` | `[1,16,160,160]` | float after Q/DQ |
| 2 | `/model.2/m.0/Add_output_0` | `[1,16,160,160]` | float Add output |

Output tensor: `/model.2/Concat_output_0`
output_shape: `[1,48,160,160]`

Post-Concat Q/DQ:

- scale: `/model.2/Concat_output_0_scale = 0.3288085460662842`
- zero_point: `/model.2/Concat_output_0_zero_point = 2`

## Decision

Concat is float-domain before post-Concat Q/DQ. Stage 12 implements it as
materialized NHWC float channel layout followed by QuantizeLinear into signed
int8 storage for `/model.2/cv2/conv/Conv`.
