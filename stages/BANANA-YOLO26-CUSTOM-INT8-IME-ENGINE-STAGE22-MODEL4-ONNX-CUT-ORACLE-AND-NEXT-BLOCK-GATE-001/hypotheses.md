# Stage22 Pre-Registered Hypotheses

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## H1_same_input_cut

Feeding the exact same full-shape `/model.4` C2f boundary tensor(s) into:

1. an ONNX Runtime CPU cut/subgraph built from the accepted Q/DQ ONNX model;
2. the integrated C++ model4 C2f runner using `Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT`;

yields bit-exact equality at `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`.

## H2_rounding_control

Equality is stable when the ambient RISC-V floating-point rounding mode is varied across RNE/RTZ/RDN/RUP/RMM, because the custom runner uses explicit RNE for quantization boundaries that require it.

## H3_fp32_fp64_reference

The ONNX cut, C++ scalar reference, and optimized runner use a clearly documented precision policy. If a float32-vs-float64 reference difference exists, Stage22 classifies it and treats ONNX Runtime CPU behavior as authority.

## H4_performance_transfer_sanity

Running the full-shape integrated runner through the real `y26_stage16_model4_c2f_run_*` / Stage21 API remains within +3% of the Stage21 accepted timing class if the same benchmark mode is used. This is a sanity check only, not a model FPS claim.
