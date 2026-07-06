# ONNX Cut Oracle Report

- model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- cut_model: `.deps/custom_int8_engine/stage22_onnx_cut/model4_c2f_cut.onnx`
- cut_model_sha256: `bde82b0130615717ffcbdbaca8fa274e5de00c111cf0b0a518023b6a674d841a`
- provider: `CPUExecutionProvider`
- onnx: `1.22.0`
- onnxruntime: `1.27.0`
- cut_input: `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output`
- cut_output: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`
- input_nhwc_bin_sha256: `e4ec6700e37e974e5bf9814b90c415169b5e514ed9554592238dd836f84fdc5b`
- output_nhwc_bin_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- cut_vs_full_model_mismatches: `0`
- cut_vs_full_model_max_abs_diff: `0`
