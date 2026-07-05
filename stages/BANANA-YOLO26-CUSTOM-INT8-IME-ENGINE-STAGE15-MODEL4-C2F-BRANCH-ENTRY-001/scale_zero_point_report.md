# Stage 15 Scale Zero Point Report

| tensor boundary | scale | zero_point_u8 | signed_storage_zero_point_s8 | note |
|---|---:|---:|---:|---|
| `/model.4/cv1/conv/Conv_output_0` | `0.0688334777951` | `163` | n/a | conv output code before SiLU |
| `/model.4/Split_output_1` | `0.0226610563695` | `12` | `-116` | first branch Conv input |
| `/model.4/m.0/cv1/conv/Conv_output_0` | `0.0436817258596` | `128` | n/a | branch Conv output code before SiLU |
| `/model.4/m.0/cv1/act/Mul_output_0` | `0.0226880107075` | `12` | `-116` | Stage 15 branch activation output |

Branch Conv weights:

- tensor: `model.4.m.0.cv1.conv.weight_quantized`
- layout in fixture: `OHWI`
- weight_zero_point: all `0`
- weight_scale_count: `16`
