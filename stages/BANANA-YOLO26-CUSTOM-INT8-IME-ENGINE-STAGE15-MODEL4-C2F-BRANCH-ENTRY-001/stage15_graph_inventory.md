# Stage 15 Graph Inventory

Model:

`/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`

SHA256:

`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`

Opset: `18`

## `/model.4` Boundary

| index | node | op | shape / contract |
|---:|---|---|---|
| `307` | `/model.4/cv1/conv/Conv` | `Conv` | input `[1,64,80,80]`, output `[1,64,80,80]`, kernel `1x1`, stride `1`, pad `0` |
| `308` | `/model.4/cv1/conv/Conv_output_0_QuantizeLinear` | `QuantizeLinear` | scale `0.06883347779512405`, zero_point_u8 `163` |
| `309` | `/model.4/cv1/conv/Conv_output_0_DequantizeLinear` | `DequantizeLinear` | dequantized float into activation |
| `310` | `/model.4/cv1/act/Sigmoid` | `Sigmoid` | float `[1,64,80,80]` |
| `311` | `/model.4/cv1/act/Mul` | `Mul` | float SiLU output `[1,64,80,80]` |
| `312` | `/model.4/Split` | `Split` | axis `1`, outputs `[1,32,80,80]` and `[1,32,80,80]` |
| `313` | `/model.4/Split_output_1_QuantizeLinear` | `QuantizeLinear` | scale `0.022661056369543076`, zero_point_u8 `12` |
| `314` | `/model.4/Split_output_1_DequantizeLinear` | `DequantizeLinear` | branch input dequantized float |
| `315` | `/model.4/m.0/cv1/conv/Conv` | `Conv` | input `[1,32,80,80]`, output `[1,16,80,80]`, kernel `3x3`, stride `1`, pad `1` |
| `316` | `/model.4/m.0/cv1/conv/Conv_output_0_QuantizeLinear` | `QuantizeLinear` | scale `0.04368172585964203`, zero_point_u8 `128` |
| `317` | `/model.4/m.0/cv1/conv/Conv_output_0_DequantizeLinear` | `DequantizeLinear` | dequantized float into branch activation |
| `318` | `/model.4/m.0/cv1/act/Sigmoid` | `Sigmoid` | float `[1,16,80,80]` |
| `319` | `/model.4/m.0/cv1/act/Mul` | `Mul` | float SiLU output `[1,16,80,80]` |
| `320` | `/model.4/m.0/cv1/act/Mul_output_0_QuantizeLinear` | `QuantizeLinear` | scale `0.022688010707497597`, zero_point_u8 `12` |
| `321` | `/model.4/m.0/cv1/act/Mul_output_0_DequantizeLinear` | `DequantizeLinear` | feeds `/model.4/m.0/cv2/conv/Conv`, deferred |

## Branch Routing

- `/model.4/Split_output_0` feeds future `/model.4/Concat`.
- `/model.4/Split_output_1` is quantized/dequantized before `/model.4/m.0/cv1/conv/Conv`.
- `/model.4/Split_output_1_DequantizeLinear_Output` also feeds future residual Add and Concat, both deferred in Stage 15.

## Stage 15 Decision

The first branch Conv/activation boundary is clear and selected.

No `/model.4/m.0/Add` or `/model.4/Concat` implementation is included in Stage 15.
