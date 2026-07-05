# Stage 14 Tensor Contract Report

## Selected Subset

`candidate_H3_model2_act_model3_act_model4_cv1_conv`

## Tensor Contracts

| tensor | compact shape | dtype/storage | scale | zero_point_u8 | producer | consumer |
|---|---|---|---:|---:|---|---|
| `/model.2/cv2/conv/Conv_output_0` corrected accumulator | `[2,2,64]` NHWC compact | `int32` | accumulator domain | n/a | Stage13 `/model.2/cv2/conv/Conv` | `/model.2/cv2/act` Q/DQ |
| `/model.2/cv2/act/Mul_output_0_QuantizeLinear_Output` | `[2,2,64]` NHWC compact | signed `int8` storage for uint8 code | `0.12448897212743759` | `2` | `/model.2/cv2/act` Q/DQ | `/model.3/conv/Conv` |
| `/model.3/conv/Conv_output_0` corrected accumulator | `[1,1,64]` NHWC compact | `int32` | accumulator domain | n/a | `/model.3/conv/Conv` | `/model.3/act` Q/DQ |
| `/model.3/act/Mul_output_0_QuantizeLinear_Output` | `[1,1,64]` NHWC compact | signed `int8` storage for uint8 code | `0.04000610113143921` | `7` | `/model.3/act` Q/DQ | `/model.4/cv1/conv/Conv` |
| `/model.4/cv1/conv/Conv_output_0` corrected accumulator | `[1,1,64]` NHWC compact | `int32` | accumulator domain | n/a | `/model.4/cv1/conv/Conv` | Stage14 output boundary |

## Conv Metadata

| node | kernel | stride | pad | Cin | Cout | weight_zero_point | correction |
|---|---|---|---|---:|---:|---|---|
| `/model.3/conv/Conv` | `3x3` | `2` | `1` | `64` | `64` | all zero | activation zero-point correction applied |
| `/model.4/cv1/conv/Conv` | `1x1` | `1` | `0` | `64` | `64` | all zero | activation zero-point correction applied |

## Layout

The runner uses compact NHWC fixtures and the existing Stage 4/5 MMT4D Conv dataflow. This is selected-subset evidence, not full-image execution.
