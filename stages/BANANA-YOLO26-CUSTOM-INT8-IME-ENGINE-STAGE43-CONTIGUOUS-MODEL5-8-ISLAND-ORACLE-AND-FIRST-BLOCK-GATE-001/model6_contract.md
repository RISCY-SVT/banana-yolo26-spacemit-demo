# Model 6 Contract

Model6 is a C2f-style block from `/model.5/act/Mul_output_0_QuantizeLinear_Output` to `/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output`.

- input/output: uint8 NCHW `1x128x40x40`;
- output scale/zero point: `0.0280377771705` / 10;
- isolated cut: 85 nodes, nine Conv, two Add, two Concat, one Split, nine Sigmoid, nine Mul, 17 QuantizeLinear, and 35 DequantizeLinear;
- cut SHA-256: `596434d51e8e4da57e2b60ee22eb85d4befa939c9f613aba96705f3810aadc20`.

No model6 implementation was authorized or written.
