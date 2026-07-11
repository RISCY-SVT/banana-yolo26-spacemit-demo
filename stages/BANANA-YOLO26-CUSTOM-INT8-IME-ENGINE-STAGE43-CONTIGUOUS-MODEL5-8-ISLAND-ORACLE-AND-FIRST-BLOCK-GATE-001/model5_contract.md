# Model 5 Contract

Graph-verified implementation boundary:

- input: `/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output`, uint8 NCHW `1x128x80x80`, scale `0.030298452824354172`, zero point 9;
- Conv: `/model.5/conv/Conv`, 3x3, stride 2, padding 1, group 1, 128 input and 128 output channels;
- output: `/model.5/act/Mul_output_0_QuantizeLinear_Output`, uint8 NCHW `1x128x40x40`, scale `0.027727888897061348`, zero point 10;
- operator mix: one Conv, four DequantizeLinear, two QuantizeLinear, Sigmoid, and Mul.

The contiguous implementation starts at the existing model4 preactivation quantized output, applies the exact model4 final SiLU/requant, executes model5 Conv, then applies exact model5 SiLU/requant. Weights are prepacked once from OIHW to OHWI/MMT4D storage.
