# Scale / Zero-Point Flow Report

## Stage 11A Flow

```text
/model.2/m.0/cv1/conv corrected int32
  input scale: /model.2/Split_output_1_scale = 0.18348428606987
  weight scale: model.2.m.0.cv1.conv.weight_scale[8]
  output scale: /model.2/m.0/cv1/conv/Conv_output_0_scale = 0.038180503994226456
  output zp: 176

requant to uint8 conv code
SiLU via boundary-specific LUT
  act scale: /model.2/m.0/cv1/act/Mul_output_0_scale = 0.012377118691802025
  act zp: 22
  int8 storage zp: -106

/model.2/m.0/cv2/conv raw dot
  input scale: 0.012377118691802025
  weight scale: model.2.m.0.cv2.conv.weight_scale[16]
  output scale: /model.2/m.0/cv2/conv/Conv_output_0_scale = 0.5276883244514465
  output zp: 179
```

## Selected Activation Mode

`A2_rvv_f32_lut` remains selected. The path uses explicit RNE conversion inherited from Stage 10 and a boundary-specific 256-code LUT for the new cv1 activation boundary.
