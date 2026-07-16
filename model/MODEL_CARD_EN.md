# YOLO26n K1X INT8 Model Card

## Supported Artifact

This release executes one prepared package:

| Field | Value |
|---|---|
| Model family | YOLO26n end-to-end detector |
| Executor contract | `K1X_INT8_V1` |
| Full-graph profile | `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` |
| Source-model SHA-256 | `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c` |
| Package manifest SHA-256 | `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be` |
| Input | RGB, fixed letterboxed `640x640` |
| Output | 300 float rows: `x1,y1,x2,y2,confidence,class` |
| Classes | COCO80 |

The `package/` directory is the complete prepared runtime model. The executor
does not parse ONNX at run time and does not use ONNX Runtime.

## Accuracy

The accepted COCO val2017 run completed 5000/5000 images. It measured
`mAP50-95=0.3707408944391919` and `mAP50=0.5258465300872381`. Its prediction
JSON SHA-256 is
`cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`.

## Limits

This package is fixed to the listed graph, input tensor, quantization contract,
and K1X executor. Camera resolution is not model resolution: every camera frame
is aspect-preserving letterboxed to 640x640. Detection reliability depends on
class, contrast, blur, occlusion, lens, distance, threshold, and object size after
letterboxing. The supplied object-size tables are measurements, not guarantees.

This engineering handoff is not production certified. No training, student
selection, or model-executor co-design was performed in Stage58.
