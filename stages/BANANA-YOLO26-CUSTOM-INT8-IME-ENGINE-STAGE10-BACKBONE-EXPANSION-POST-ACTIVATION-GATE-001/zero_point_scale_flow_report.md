# Zero Point and Scale Flow Report

## Existing Stage 9 Boundaries

- Images Q/DQ -> Conv0: input scale `0.00392156885937`, zp `0`, signed storage `-128`.
- Conv0 output: scale `0.620968401432`, zp `128`; Act0 output scale `0.311162889004`, zp `1`.
- Conv1 output: scale `1.1142437458`, zp `122`; Act1 output scale `0.582500994205`, zp `0`.
- Conv2 output: scale `0.836495816708`, zp `155`.

## New Stage 10 Boundary

Conv2 corrected int32 -> conv output code -> SiLU LUT -> Split output 1 signed storage:

- conv2 output scale: `0.8364958167076111`
- conv2 output zero point: `155`
- split output 1 scale: `0.18348428606987`
- split output 1 zero point: `2`
- branch0 input storage zero point: `-126`

## Branch Conv

Branch node: `/model.2/m.0/cv1/conv/Conv`

- input scale: `0.18348428606987`
- input zero point: `2`
- weights: signed int8, per-output-channel scales, zero-points all `0`
- output scale: `0.038180503994226456`
- output zero point: `176`
- output boundary: corrected int32, no requant after branch in Stage 10

## Correction Formula

Raw kernels compute signed-storage dot products. For uint8 activation code `q`, signed storage is `q - 128`; selected branch input zp is `2`, so the correction offset is `128 - 2 = 126`:

`corrected = raw_dot + 126 * sum(weights_oc) + bias_oc`

This is implemented through `y26_conv2d_apply_u8_as_s8_correction_nhwc`.
