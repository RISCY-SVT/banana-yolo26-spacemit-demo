# Stage 14 Next Block Graph Inventory

model: `/data/banana-yolo26-spacemit-demo/.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx`
model_sha256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`

## New Nodes

| node | op | input | output | shape | contract |
|---|---|---|---|---|---|
| `/model.2/cv2/act/Sigmoid` | `Sigmoid` | `/model.2/cv2/conv/Conv_output_0_DequantizeLinear_Output` | `/model.2/cv2/act/Sigmoid_output_0` | compact `[1,64,2,2]` in fixture | float SiLU component |
| `/model.2/cv2/act/Mul` | `Mul` | conv dequant + sigmoid | `/model.2/cv2/act/Mul_output_0` | compact `[1,64,2,2]` in fixture | float SiLU output |
| `/model.3/conv/Conv` | `Conv` | `/model.2/cv2/act/Mul_output_0_DequantizeLinear_Output` | `/model.3/conv/Conv_output_0` | compact NHWC `[2,2,64] -> [1,1,64]` | 3x3 stride2 pad1 |
| `/model.3/act/Sigmoid` | `Sigmoid` | `/model.3/conv/Conv_output_0_DequantizeLinear_Output` | `/model.3/act/Sigmoid_output_0` | compact `[1,64,1,1]` in fixture | float SiLU component |
| `/model.3/act/Mul` | `Mul` | conv dequant + sigmoid | `/model.3/act/Mul_output_0` | compact `[1,64,1,1]` in fixture | float SiLU output |
| `/model.4/cv1/conv/Conv` | `Conv` | `/model.3/act/Mul_output_0_DequantizeLinear_Output` | `/model.4/cv1/conv/Conv_output_0` | compact NHWC `[1,1,64] -> [1,1,64]` | 1x1 stride1 |

The full ONNX node dump is preserved in:
`/data/ncnn-logs/ai-team/2026-07-05_07-43-30/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE14-NEXT-C2F-BLOCK-EXPANSION-001/run_logs/onnx_graph_inventory_exact_model2_4.log`.
