# Zero-Point and Scale Flow Report

Selected subset: `candidate_C_block0_silu_model1_conv`

## Conv0

Input:

```text
images_scale = 0.00392156885937
images_zero_point_u8 = 0
runtime_storage = int8(q_u8 - 128)
runtime_storage_zero_point_s8 = -128
```

Weights:

```text
model.0.conv.weight_quantized: int8 per-output-channel
model.0.conv.weight_zero_point: all 0
```

Correction:

```text
raw = sum((q_u8 - 128) * w_s8)
corrected = raw + (128 - images_zero_point_u8) * sum(w_s8) + bias_i32
```

## Conv0 Output and Activation

Conv0 corrected int32 is dequantized with per-channel accumulator scale:

```text
conv0_acc_scale[oc] = images_scale * model.0.conv.weight_scale[oc]
```

Then ONNX Q/DQ and SiLU are applied as scalar float fallback:

```text
conv0_output_scale = 0.620968401432
conv0_output_zero_point_u8 = 128
act0_output_scale = 0.311162889004
act0_output_zero_point_u8 = 1
```

The activation output is stored for Conv1 as:

```text
conv1_input_s8 = int8(act_q_u8 - 128)
conv1_activation_zero_point_u8 = 1
conv1_input_storage_zero_point_s8 = -127
```

## Conv1

Weights:

```text
model.1.conv.weight_quantized: int8 per-output-channel
model.1.conv.weight_zero_point: all 0
```

Correction:

```text
raw = sum((act_q_u8 - 128) * w_s8)
corrected = raw + (128 - act0_output_zero_point_u8) * sum(w_s8) + bias_i32
```

Conv1 output Q/DQ and Conv1 SiLU are not integrated in Stage 6.

