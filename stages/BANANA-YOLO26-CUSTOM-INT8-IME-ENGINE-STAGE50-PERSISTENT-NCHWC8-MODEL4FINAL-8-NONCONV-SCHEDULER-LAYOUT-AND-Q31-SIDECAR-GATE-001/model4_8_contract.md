# Model4-final to model8 contract

- Contract: `K1X_INT8_V1` / `K1X_INT8_V1_GENERAL`
- Layout: `NCHWc8_SPATIAL_INNER_V1`
- Model SHA-256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- Input: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`, uint8 logical NCHW `1x128x80x80`
- Output: `/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output`, uint8 logical NCHW `1x256x20x20`
- Package manifest SHA-256: `2dbdbd18abe1ba126f12246b82c25821b9f74eb0ee9c324cb30aaaa062f64527`

The package contains 32 tensor boundaries and 29 operations. Conv, activation, Add/residual, Split/view, and Concat rescale/placement semantics are package-defined integer operations. The timed custom region has no ORT call, float Q/DQ materialization, or internal NCHW/NCHWc8 conversion.
