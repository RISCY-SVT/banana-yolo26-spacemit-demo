# Model16 Quantized Contract

Graph-valid QuantizeLinear outputs define the implementation-oriented contract:

```text
input name: /model.15/Concat_output_0_QuantizeLinear_Output
input dtype: uint8
input shape/layout: 1x256x80x80 NCHW
input scale: 0.0302984528244
input zero point: 9

output name: /model.16/cv2/act/Mul_output_0_QuantizeLinear_Output
output dtype: uint8
output shape/layout: 1x64x80x80 NCHW
output scale: 0.021654414013
output zero point: 13

cut SHA256: 5ee79a3551b54667db9e34842bbcf19b0e25d909edb42f39c75c71c2c9e58cb0
```

Input NPY SHA-256 is `808923db45cdc12d574dfdac69a9164c944c2b439682972806f8a304703f0c4e`; output NPY SHA-256 is `e37aa4bc9a735e260df51c561545669c61db270a0dbc43131d7b9bd27aaabfc1`.

This contract avoids an unnecessary float round trip and is the preferred future custom-block test surface. No dtype was inferred from a tensor name alone, and no optimized model16 code was implemented.
