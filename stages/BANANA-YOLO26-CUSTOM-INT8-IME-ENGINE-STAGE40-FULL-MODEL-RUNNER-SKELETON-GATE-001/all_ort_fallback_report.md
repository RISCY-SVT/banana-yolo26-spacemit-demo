# Stage40 Block Cut Skeleton Report

- model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- provider: `CPUExecutionProvider`
- model4_cut_input: `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output`
- model4_cut_output: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`
- all_ort_status: `pass`
- custom_model4_boundary_status: `pass`
- custom_model4_final_status: `pass`

| comparison | status | mismatches | max_abs_diff |
|---|---:|---:|---:|
| prefix_vs_full_model4_input | pass | 0 | 0 |
| all_ort_model4_vs_full_model4_output | pass | 0 | 0 |
| all_ort_final_vs_full_reference | pass | 0 | 0 |
| custom_model4_output_vs_full_model4_output | pass | 0 | 0 |
| custom_model4_skeleton_final_vs_full_reference | pass | 0 | 0 |

Timing is skeleton profiling only, not model FPS.
