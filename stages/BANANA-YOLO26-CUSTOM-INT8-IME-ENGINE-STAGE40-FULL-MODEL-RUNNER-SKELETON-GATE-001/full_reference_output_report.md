# Full ONNX Runtime CPU Reference Report

- model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- provider: `CPUExecutionProvider`
- input_name: `images`
- input_shape: `[1, 3, 640, 640]`
- input_mode: `synthetic_seeded`
- outputs: `output0, /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output, /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`
