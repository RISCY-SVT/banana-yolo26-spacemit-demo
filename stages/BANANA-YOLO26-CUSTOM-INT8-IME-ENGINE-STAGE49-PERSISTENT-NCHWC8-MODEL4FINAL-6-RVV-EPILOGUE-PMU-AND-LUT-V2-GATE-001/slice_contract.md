# Persistent slice contract

- Contract: `K1X_INT8_V1`, general profile
- Logical input: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`, uint8 NCHW `1x128x80x80`, represented resident signed NCHWc8
- Logical output: `/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output`, uint8 NCHW `1x128x40x40`, represented resident signed NCHWc8
- Operations: model4 final LUT, model5 Conv+activation, complete model6 C2f integer schedule
- Split: view; Concat: package-defined direct placement/rescale LUT; Add: package-defined exact 256x256 LUT; Conv/activation: exact integer assets
- No implicit float fallback is permitted.
