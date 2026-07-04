# Boundary LUT Oracle Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`

## Boundary

`/model.2/cv1/conv/Conv_output_0` Q/DQ + SiLU -> `/model.2/Split_output_1` Q/DQ.

## ONNX Runtime 256-code Oracle

- model_path: `/data/ncnn-logs/ai-team/2026-07-04_15-15-28/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001/run_logs/stage10_conv2_to_split_lut.onnx`
- model_sha256: `6f1e4def6288f67b3e05e8cba294f0f2cbb5a9bf13e239fab74c7e14e57667ef`
- providers: `CPUExecutionProvider`
- conv_scale: `0.8364958167076111`
- conv_zero_point: `155`
- split_output1_scale: `0.18348428606987`
- split_output1_zero_point: `2`
- mismatches: `0`
- max_abs_diff_u8: `0`

## Decision

The Stage 10 internal LUT for the new boundary matches the ONNX Runtime standalone Q/DQ+SiLU+Q oracle for all 256 input codes.
