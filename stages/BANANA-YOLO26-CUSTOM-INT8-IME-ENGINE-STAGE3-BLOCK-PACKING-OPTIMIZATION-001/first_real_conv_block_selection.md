# First Real Conv Block Selection

Accepted model:

`/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`

## Graph Summary

- Nodes: 1069
- Conv nodes: 102
- Q/DQ graph: `QuantizeLinear` 206, `DequantizeLinear` 410
- First implementation remains traditional/trunk-first, not e2e head-first.

## Selected / Documented Nodes

### First Real Conv Node

- Node: `/model.0/conv/Conv`
- Type: `Conv`
- Input: `images_DequantizeLinear_Output`
- Output: `/model.0/conv/Conv_output_0` `[1,16,320,320]`
- Kernel: `3x3`
- Stride: `2x2`
- Pads: `[1,1,1,1]`
- Group: `1`
- Cin/Cout: `3/16`
- Activation quant: `uint8`, scale `0.003921568859368563`, zero-point `0`
- Weight quant: `int8`, per-output-channel scale `[16]`, zero-point all `0`
- Correction required: yes, because activation storage is `uint8` and IME consumes signed `int8`.
- Stage 3 implementation target: documented but not used as primary benchmark because `Cin=3` is tail-heavy and not representative for the first MMT4D packing optimization.

### Simple 1x1 Real Conv Target

- Node: `/model.2/cv1/conv/Conv`
- Input tensor: `/model.1/act/Mul_output_0_DequantizeLinear_Output`
- Quant input tensor: `/model.1/act/Mul_output_0_QuantizeLinear_Output`
- Output tensor: `/model.2/cv1/conv/Conv_output_0`
- Shape: input `[1,32,160,160]`, output `[1,32,160,160]`
- Kernel/stride/pad: `1x1`, `1x1`, `[0,0,0,0]`
- Activation quant: `uint8`, scale `0.5825009942054749`, zero-point `0`
- Weight quant: `int8`, per-output-channel scale `[32]`, zero-point all `0`
- Selected for Stage 3 real Conv1x1 fixture and packing benchmark.

### Simple 3x3 Real Conv Target

- Node: `/model.2/m.0/cv1/conv/Conv`
- Input tensor: `/model.2/Split_output_1_DequantizeLinear_Output`
- Quant input tensor: `/model.2/Split_output_1_QuantizeLinear_Output`
- Output tensor: `/model.2/m.0/cv1/conv/Conv_output_0`
- Shape: input `[1,16,160,160]`, output `[1,8,160,160]`
- Kernel/stride/pad: `3x3`, `1x1`, `[1,1,1,1]`
- Activation quant: `uint8`, scale `0.18348428606987`, zero-point `2`
- Weight quant: `int8`, per-output-channel scale `[8]`, zero-point all `0`
- Selected for Stage 3 real Conv3x3 fixture and packing benchmark.

## Decision

Stage 3 selected real Conv1x1 and Conv3x3 nodes that are early, tractable, non-grouped, and representative of the MMT4D path. `/model.0/conv/Conv` is recorded as the first real Conv node, but the Stage 3 optimization target is the simple 1x1/3x3 pair above.
