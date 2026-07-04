# Tensor Contract Report

## Selected Stage 11A Tensors

| tensor_name | shape | dtype/domain | producer | consumer | scale | zero_point | signedness/storage | layout | workspace |
|---|---:|---|---|---|---:|---:|---|---|---|
| `/model.2/m.0/cv1/conv/Conv_output_0` | `[1,8,160,160]` real, `[2,2,8]` fixture NHWC | int32 accumulator before Q | `/model.2/m.0/cv1/conv/Conv` | Q/DQ + SiLU | `0.038180503994226456` | `176` | uint8 code after requant, int8 storage `q-128` | NHWC in runner | `branch0_i32` |
| `/model.2/m.0/cv1/act/Mul_output_0` | `[1,8,160,160]` real, `[2,2,8]` fixture NHWC | quantized activation | SiLU | `/model.2/m.0/cv2/conv/Conv` | `0.012377118691802025` | `22` | int8 storage zp `-106` | NHWC | `branch0_act_s8` |
| `model.2.m.0.cv2.conv.weight_quantized` | `[16,8,3,3]` OIHW, stored as OHWI fixture | int8 weight | initializer | `/model.2/m.0/cv2/conv/Conv` | per-output-channel, count `16` | all `0` | signed symmetric | prepacked MMT4D B | persistent prepack |
| `/model.2/m.0/cv2/conv/Conv_output_0` | `[1,16,160,160]` real, `[2,2,16]` fixture NHWC | corrected int32 output | `/model.2/m.0/cv2/conv/Conv` | Stage 11 output boundary | `0.5276883244514465` | `179` | uint8 code after future requant | NHWC | output buffer |

## Correction

Weights are symmetric signed with zero-point `0`. Activation storage is signed `q_u8 - 128`, so Conv correction remains:

`corrected = raw_dot + (128 - activation_zero_point_u8) * sum(weights_oc) + bias_oc`

For Stage 11 branch cv2:

`activation_zero_point_u8 = 22`, so the additive correction coefficient is `106`.
