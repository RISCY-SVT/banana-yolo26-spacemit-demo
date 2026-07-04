# Boundary LUT Oracle Report

new_boundary: `/model.2/m.0/cv1/conv/Conv_output_0` -> `/model.2/m.0/cv1/act/Mul_output_0`

## Inputs

- conv_output_scale: `0.038180503994226456`
- conv_output_zero_point_u8: `176`
- act_output_scale: `0.012377118691802025`
- act_output_zero_point_u8: `22`

## ONNX Runtime 256-Code Oracle

- oracle_model: `/data/ncnn-logs/ai-team/2026-07-04_19-42-26/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE11-BRANCH-BLOCK-EXPANSION-001/run_logs/stage11_branch0_to_act_lut.onnx`
- oracle_model_sha256: `8af7be0904e1e61603d82b59bffd9dab421f122cc80fc024a8242eb955f6ebc7`
- mismatches: `0`
- max_abs_diff_u8: `0`

The accepted Stage 11 runner does not reuse Stage 10 LUT values for this boundary. It generates and tests the new boundary-specific LUT.
