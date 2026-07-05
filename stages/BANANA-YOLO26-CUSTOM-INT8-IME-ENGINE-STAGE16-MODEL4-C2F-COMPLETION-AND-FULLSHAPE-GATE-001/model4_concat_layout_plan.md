# Model4 Concat Layout Plan

`/model.4/Concat` is a float-domain channel concat over ONNX NCHW axis `1`. The Stage16 compact runner represents tensors in NHWC fixture layout and maps channel spans exactly:

- channels `[0,32)`: `/model.4/Split_output_0`
- channels `[32,64)`: `/model.4/Split_output_1_DequantizeLinear_Output`
- channels `[64,96)`: `/model.4/m.0/Add_output_0`

Accepted Stage16 candidate materializes post-Concat signed int8 storage after applying post-Concat Q/DQ (`scale=0.03763701394200325`, `zero_point_u8=15`). This avoids claiming an integer-domain concat shortcut and preserves ONNX float merge behavior.

Future work: a view/span concat packer may be revisited only with exact post-Concat Q/DQ and `/model.4/cv2/conv/Conv` oracle equivalence.
