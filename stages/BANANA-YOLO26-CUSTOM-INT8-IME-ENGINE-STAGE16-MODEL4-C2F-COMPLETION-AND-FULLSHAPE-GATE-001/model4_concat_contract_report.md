# Model4 Concat Contract Report

node: `/model.4/Concat`
axis: `1` in ONNX NCHW, represented as channel concatenation in NHWC runner fixtures
inputs:
- channels `[0,32)`: `/model.4/Split_output_0`, float-domain direct Split output
- channels `[32,64)`: `/model.4/Split_output_1_DequantizeLinear_Output`, float-domain Q/DQ output
- channels `[64,96)`: `/model.4/m.0/Add_output_0`, float-domain Add output
post_concat_qdq_scale: `0.037637013942`
post_concat_qdq_zero_point_u8: `15`
accepted Stage 16 compact path: materialize post-Concat signed int8 storage for `/model.4/cv2/conv/Conv`
