# Runner Boundary Contracts

## Full Reference

```text
input:  images float32 NCHW 1x3x640x640
output: output0 float32 1x300x6
```

## Prefix Cut

```text
input:  images float32 NCHW 1x3x640x640
output: /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output uint8 NCHW 1x64x80x80
```

## Custom Model4 Runner Handoff

```text
input file:  model4_cv1_conv_q_u8_nhwc.bin
input shape: 1x80x80x64 NHWC uint8
output file: custom_model4_cv2_q_u8_nhwc.bin
output shape: 1x80x80x128 NHWC uint8
```

## Suffix Cut

```text
input:  /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output uint8 NCHW 1x128x80x80
output: output0 float32 1x300x6
```

The Python skeleton transposes custom NHWC output back to ONNX NCHW before running the suffix cut.
