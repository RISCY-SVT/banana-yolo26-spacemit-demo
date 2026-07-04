# Oracle Extraction Report

classification: `pass`

## Tooling

- venv: `.deps/custom_int8_engine/venv-stage7-onnx`
- Python packages: `onnx 1.22.0`, `onnxruntime 1.27.0`, `numpy 2.5.0`
- runtime dependency added: `false`
- xslim_used: `false`

## Model

- path: `/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
- SHA256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- oracle model copy: `/data/banana-yolo26-spacemit-demo/.deps/custom_int8_engine/stage7_oracle/2026-07-04_08-03-01/stage7_backbone_subset_outputs.onnx`

## Selected Nodes

- `images_QuantizeLinear` `QuantizeLinear`
- `images_DequantizeLinear` `DequantizeLinear`
- `/model.0/conv/Conv` `Conv`
- `/model.0/conv/Conv_output_0_QuantizeLinear` `QuantizeLinear`
- `/model.0/conv/Conv_output_0_DequantizeLinear` `DequantizeLinear`
- `/model.0/act/Sigmoid` `Sigmoid`
- `/model.0/act/Mul` `Mul`
- `/model.0/act/Mul_output_0_QuantizeLinear` `QuantizeLinear`
- `/model.0/act/Mul_output_0_DequantizeLinear` `DequantizeLinear`
- `/model.1/conv/Conv` `Conv`
- `/model.1/conv/Conv_output_0_QuantizeLinear` `QuantizeLinear`
- `/model.1/conv/Conv_output_0_DequantizeLinear` `DequantizeLinear`
- `/model.1/act/Sigmoid` `Sigmoid`
- `/model.1/act/Mul` `Mul`
- `/model.1/act/Mul_output_0_QuantizeLinear` `QuantizeLinear`
- `/model.1/act/Mul_output_0_DequantizeLinear` `DequantizeLinear`
- `/model.2/cv1/conv/Conv` `Conv`
- `/model.2/cv1/conv/Conv_output_0_QuantizeLinear` `QuantizeLinear`
- `/model.2/cv1/conv/Conv_output_0_DequantizeLinear` `DequantizeLinear`

## Quantization

- images scale/zp: `0.003921568859368563` / `0`
- conv0 output scale/zp: `0.6209684014320374` / `128`
- act0 output scale/zp: `0.31116288900375366` / `1`
- conv1 output scale/zp: `1.114243745803833` / `122`
- act1 output scale/zp: `0.5825009942054749` / `0`
- conv2 output scale/zp: `0.8364958167076111` / `155`

## Cases

| case | input crop | conv2 shape | conv0 max abs diff | conv1 max abs diff | conv2 max abs diff | expected conv2 SHA256 |
|---|---:|---|---:|---:|---:|---|
| `synthetic_seeded` | `8x8` | `[2, 2, 32]` | `3.814697265625e-06` | `1.1444091796875e-05` | `5.7220458984375e-06` | `fd5eadace7c7ab7c984b3106523f78ad6b14bde7f1cfb266ec04bcdfe3581b0e` |
| `synthetic_gradient` | `8x8` | `[2, 2, 32]` | `1.9073486328125e-06` | `3.814697265625e-06` | `1.7285346984863281e-06` | `286684755909d672c0f04cd95501cf7ccc8fda1826ef4211cd6b73c208d43f23` |


Fixture header: `custom_int8_engine/tests/stage7_backbone_subset_fixture.h`
Fixture SHA256: `86a96603ff22efa600518791c05707d79d52add131b3858c82ce761eba76b505`
Large dumps: `.deps/custom_int8_engine/stage7_oracle/2026-07-04_08-03-01/` (not tracked).
