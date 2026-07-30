# XSlim issue 45 decision

## ReduceMax matrix

| Lane | opset-18 two-input | opset-13 attribute control | Conv + opset-18 ReduceMax |
|---|---|---|---|
| official XSlim 2.1.1 | still broken | still broken after conversion to opset 24 | still broken |
| vendor ref `9a33f2f` | fixed, exact | fixed, exact | fixed; quantized output max error 0.0149146 |

The official release converts each tiny graph to opset 24 and then calls a
`ReduceMax_forward` implementation that destructures exactly one runtime
input. The converted graph supplies the axes as a second input and raises
`ValueError: too many values to unpack`.

The vendor-referenced commit routes ReduceMax through `_get_reduce_inputs`.
All three generated models pass ONNX checking and host execution. Pure
ReduceMax controls are exact.

## YOLO26 implication

The six-output split truncates the graph before the YOLO post-processing
ReduceMax, so both XSlim lanes can quantize the inference subgraph despite the
official release defect. This is classified as `bypassed-by-truncation` for
the split workflow, not as an official 2.1.1 ReduceMax fix.

Direct E2E confirms both sides of that boundary. Official 2.1.1 fails during
metadata tracing at `/model.23/ReduceMax`. The vendor-reference commit emits a
valid signed-QDQ model, but all 100 host holdout images have score channels
collapsed to zero. The direct artifact is rejected before board execution and
cannot replace the vendor split workflow.
