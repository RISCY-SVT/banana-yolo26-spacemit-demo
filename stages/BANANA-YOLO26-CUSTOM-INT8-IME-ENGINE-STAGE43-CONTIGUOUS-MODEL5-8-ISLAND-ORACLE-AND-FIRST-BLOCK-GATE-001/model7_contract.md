# Model 7 Contract

Model7 is the next downsampling Conv block.

- input: `/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output`, uint8 NCHW `1x128x40x40`, scale `0.0280377771705`, zero point 10;
- output: `/model.7/act/Mul_output_0_QuantizeLinear_Output`, uint8 NCHW `1x256x20x20`, scale `0.0341685414314`, zero point 8;
- isolated cut: nine nodes with one Conv and SiLU/QDQ handoff;
- cut SHA-256: `1f162b5ee6cf39ccd868fa52e771f3f0ff834e3507efc27c0e700f3ced6b3f19`.

No model7 implementation was authorized or written.
