# Model 8 Contract

Model8 is a C2f-style block from `/model.7/act/Mul_output_0_QuantizeLinear_Output` to `/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output`.

- input/output: uint8 NCHW `1x256x20x20`;
- output scale/zero point: `0.0326153002679` / 9;
- isolated cut: 85 nodes with nine Conv, two Add, two Concat, one Split, nine Sigmoid, nine Mul, 17 QuantizeLinear, and 35 DequantizeLinear;
- cut SHA-256: `fb20321de452c2a27ef945fffb2738a2f29e7ddf97140523f120574641ecd43c`.

No model8 implementation was authorized or written.
