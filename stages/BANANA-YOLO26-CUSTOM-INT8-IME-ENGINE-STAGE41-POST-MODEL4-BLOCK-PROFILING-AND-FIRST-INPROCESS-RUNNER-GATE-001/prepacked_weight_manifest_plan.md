# Prepacked Weight Manifest Plan

For every future custom block, record:

```text
onnx_weight_tensor_name
source_sha256_or_model_sha256
shape
dtype
scale/zp metadata
packed_layout
packed_bytes
kernel_owner
compatible input layout
```

Weights must be packed at prepare/init time, not per inference hot loop.
