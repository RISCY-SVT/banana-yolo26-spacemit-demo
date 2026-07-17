# Source Model Provenance

| Field | Recorded value |
|---|---|
| Filename | `manual_e2e_rep_conv_matmul_qdq.onnx` |
| SHA-256 | `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c` |
| Controlled local source | `.deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx` |
| First accepted full-package use | Stage52 |
| Runtime contract | `K1X_INT8_V1` |
| Full-graph profile | `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` |
| Export creator | Not established by the retained record |
| Exact export command | Not established by the retained record |

Stage52 command evidence shows this exact artifact as the input to
`custom_int8_engine/tools/stage52_full_package.py`. Stage59 located the same
local artifact and reverified its SHA-256; it did not download, regenerate, or
substitute a model.
