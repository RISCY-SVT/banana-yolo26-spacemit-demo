# Graph Inventory

model: `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`

```text
input: images float32 1x3x640x640
output: output0 float32 1x300x6
nodes: 1069
initializers: 1024
```

## Operator Counts

| op_type | count |
|---|---:|
| DequantizeLinear | 410 |
| QuantizeLinear | 206 |
| Conv | 102 |
| Mul | 93 |
| Sigmoid | 88 |
| Constant | 47 |
| Concat | 26 |
| Add | 23 |
| Reshape | 14 |
| Split | 12 |
| Transpose | 5 |
| MatMul | 4 |
| Gather | 4 |
| MaxPool | 3 |
| Shape | 3 |
| Unsqueeze | 3 |
| Cast | 3 |
| other scalar/postprocess ops | 29 |

## Stage40 Cut Points

| cut | input | output | nodes |
|---|---|---|---:|
| prefix | `images` | `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output` | 73 |
| model4 | `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output` | `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output` | 31 |
| suffix | `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output` | `output0` | 966 |

The immediate post-model4 graph starts at `/model.5/conv/Conv`, then enters `/model.6/...`.
