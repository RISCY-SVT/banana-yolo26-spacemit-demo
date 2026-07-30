# Source model provenance

## Selected model

The canonical floating-point input is:

```text
filename: yolo26n_640_e2e_fp32.onnx
SHA-256: d71286588abe691ede49faa5ca9a471b7e9e5257669953ee59abbc2e9d115fc2
input: float32 images, 1x3x640x640
output: float32 output0, 1x300x6
IR: 8
opset: 18
nodes: 453
initializers: 204
```

The selected artifact is byte-identical to the accepted
`ultralytics_latest/end2end_true640/yolo26n.onnx` copy in the existing
read-only model evidence. It was not downloaded or re-exported in Stage64.

## Eligibility

The graph contains no `QuantizeLinear`, `DequantizeLinear`, or QLinear
operators and contains all six vendor-named truncation tensors. Learned weight
lineage is therefore anchored in a floating-point graph suitable for XSlim
static PTQ.

The Stage63 model
`manual_e2e_rep_conv_matmul_qdq.onnx`, SHA-256
`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`,
already contains Q/DQ and remains only the historical unsupported-format
negative control. Stage64 never feeds it back through XSlim.

## Distribution

The full model, generated quantized models, and trained initializer payloads
remain outside Git and outside the result packet. Stage64 does not authorize
external model publication or change existing license conclusions.
