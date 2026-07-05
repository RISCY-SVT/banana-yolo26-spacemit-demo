# Stage 16 Model4 C2f Oracle Report

model: `/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
provider: `CPUExecutionProvider` for 256-code activation micro-oracles
selected_subset: `candidate_J_model4_c2f_complete_compact`

## Compact Fixture Checksums

seeded_branch1_i32_checksum: `237176`
seeded_concat_s8_checksum: `-8703`
seeded_model4_cv2_i32_checksum: `-143848`
gradient_branch1_i32_checksum: `242635`
gradient_concat_s8_checksum: `-8668`
gradient_model4_cv2_i32_checksum: `-66786`

## Contract

`/model.4/m.0/Add` and `/model.4/Concat` are float-domain operations in the accepted Q/DQ graph.
The compact fixture preserves this by computing float Split0, Q/DQ float Split1, float branch cv2 SiLU,
float Add, float Concat, post-Concat Q/DQ, then corrected int32 `/model.4/cv2/conv/Conv`.
