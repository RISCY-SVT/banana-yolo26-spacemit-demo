# Stage41 Suffix Inventory Report

- model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- suffix_blocks: `19`
- cumulative_cuts_extracted: `19`
- input_boundary: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`
- cumulative cut timings must be interpreted as suffix-prefix timings from the same input boundary, not isolated block runtimes.
