# First Block Selection

Selected block id: `block0_conv_only`

Selected node:

- node: `/model.0/conv/Conv`
- op type: `Conv`
- input tensor: `images_DequantizeLinear_Output`
- weight tensor: `model.0.conv.weight_DequantizeLinear_Output`
- bias tensor: `model.0.conv.bias`
- output tensor: `/model.0/conv/Conv_output_0`
- model input shape: `[1,3,640,640]`
- Conv output shape: `[1,16,320,320]`
- kernel: `3x3`
- stride: `2x2`
- pads: `[1,1,1,1]`
- group: `1`
- Cin: `3`
- Cout: `16`

## Quantization Metadata

- activation Q tensor: `images_QuantizeLinear_Output`
- activation scale: `0.00392156885937`
- activation zero-point: `0` as `uint8`
- engine storage convention: `q_u8 - 128`, so input storage zero-point is `-128` as `int8`
- weight Q tensor: `model.0.conv.weight_quantized`
- weight shape: `[16,3,3,3]`
- weight zero-points: per-output-channel `int8`, all `0`
- weight scale: per-output-channel `float32`, min `0.03208913`, max `0.18584141`
- bias Q tensor: `model.0.conv.bias_quantized`
- output Q tensor: `/model.0/conv/Conv_output_0_QuantizeLinear_Output`
- output scale: `0.620968401432`
- output zero-point: `128`

## Boundary Decision

Chosen boundary: `block0_conv_only`.

Immediate downstream path:

1. `/model.0/conv/Conv_output_0_QuantizeLinear`
2. `/model.0/conv/Conv_output_0_DequantizeLinear`
3. `/model.0/act/Sigmoid`
4. `/model.0/act/Mul`

The downstream activation is SiLU expressed as `Sigmoid` + `Mul`, so it is not treated as a trivial local int32 boundary in Stage 5. Stage 5 integrates only the first Conv block, persistent weight prepack, reusable workspace, raw dot, zero-point correction, and corrected int32 output. Activation, output requant, and multi-node block scheduling are deferred.

No XSlim path was used.
