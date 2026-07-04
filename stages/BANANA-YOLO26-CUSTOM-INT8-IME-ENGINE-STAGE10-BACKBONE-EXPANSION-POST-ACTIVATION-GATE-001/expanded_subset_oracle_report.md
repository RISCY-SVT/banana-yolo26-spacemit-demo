# Stage 10 Fixture Oracle Metadata

model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
selected_subset: `candidate_E_branch1_stage9_split_model2_m0_cv1_conv`
selected_new_boundary: `/model.2/Split` output 1 -> `/model.2/m.0/cv1/conv/Conv`

## Quantization

- conv2_output_scale: `0.8364958167076111`
- conv2_output_zero_point: `155`
- split_output1_scale: `0.18348428606987`
- split_output1_zero_point: `2`
- branch0_output_scale: `0.038180503994226456`
- branch0_output_zero_point: `176`
- branch0_weight_scale_count: `8`
- branch0_weight_zero_points: all `0`

## Small Fixture Checksums

- seeded_conv2_act_sum: `-13566`
- seeded_split_output1_sum: `-6907`
- seeded_branch0_i32_sum: `-31641`
- gradient_conv2_act_sum: `-14833`
- gradient_split_output1_sum: `-7367`
- gradient_branch0_i32_sum: `92024`

## Scope

The generated fixture is a compact deterministic scalar oracle using real ONNX
weights, scales, zero-points, and Stage 7 Conv2 corrected-int32 fixture tensors.
Large full-shape tensor dumps are intentionally not committed.
