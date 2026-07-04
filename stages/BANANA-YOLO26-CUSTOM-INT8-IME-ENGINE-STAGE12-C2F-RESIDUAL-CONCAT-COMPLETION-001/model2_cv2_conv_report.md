# Model2 Cv2 Conv Report

node: `/model.2/cv2/conv/Conv`
input_boundary: post-Concat signed int8 storage
output_boundary: corrected int32 output of `/model.2/cv2/conv/Conv`

## Contract

- compact fixture input shape NHWC: `[2,2,48]`
- full bench input shape NHWC: `[160,160,48]`
- compact fixture output shape NHWC: `[2,2,64]`
- full bench output shape NHWC: `[160,160,64]`
- kernel: `1x1`
- stride: `1x1`
- padding: `0`
- input scale: `0.3288085460662842`
- input zero_point_u8: `2`
- input storage zero_point_s8: `-126`
- output scale: `0.4553883671760559`
- output zero_point_u8: `186`
- weight shape OIHW: `[64,48,1,1]`
- weight zero-points: all `0`
- weight scales: per-output-channel, count `64`

## Correctness

- Host CTest: `29/29` pass
- Board CPU0/1/2/3: `model2_cv2_mismatches=0`

## Timing

CPU0 full-shape Stage 12 IME A2:

- `model2_cv2_conv_us=50056.6`
- `conv_share_pct=47.0785` for the whole selected subset
