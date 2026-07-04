# Branch Block Selection

selected_subset: `candidate_F_model2_m0_cv1_act_cv2_conv`

## Stage 10 Input Boundary

Replayed Stage 10 subset:

- `/model.0/conv/Conv`
- `/model.0/act/Sigmoid` + `/model.0/act/Mul`
- `/model.1/conv/Conv`
- `/model.1/act/Sigmoid` + `/model.1/act/Mul`
- `/model.2/cv1/conv/Conv`
- `/model.2/cv1/act/Sigmoid` + `/model.2/cv1/act/Mul`
- `/model.2/Split`
- `/model.2/m.0/cv1/conv/Conv`

## Stage 11A Added Boundary

- `/model.2/m.0/cv1/conv/Conv_output_0` Q/DQ
- `/model.2/m.0/cv1/act/Sigmoid`
- `/model.2/m.0/cv1/act/Mul`
- `/model.2/m.0/cv1/act/Mul_output_0` Q/DQ
- `/model.2/m.0/cv2/conv/Conv`

Output boundary:

- corrected int32 output of `/model.2/m.0/cv2/conv/Conv`

## Deferred

- `/model.2/m.0/cv2/act/Sigmoid` + `/model.2/m.0/cv2/act/Mul`
- `/model.2/m.0/Add`
- `/model.2/Concat`
- `/model.2/cv2/conv/Conv`

Residual Add is deferred because ONNX represents `/model.2/m.0/Add` as a float-domain Add between `/model.2/Split_output_1_DequantizeLinear_Output` and `/model.2/m.0/cv2/act/Mul_output_0`.
