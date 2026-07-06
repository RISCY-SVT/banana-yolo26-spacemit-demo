# Cross-Track `/data/ncnn` Note

Stage21 did not inspect deeply, clean, modify, or depend on `/data/ncnn`.

At Stage21 preflight, `/data/ncnn` had unrelated dirty files:

```text
 M src/layer/riscv/convolution_1x1_int8_xsmtvdot.S
 M src/layer/riscv/convolution_1x1_int8_xsmtvdot.cpp
 M src/layer/riscv/convolution_1x1_int8_xsmtvdot.h
```

These changes are outside the Stage21 product repo scope.
