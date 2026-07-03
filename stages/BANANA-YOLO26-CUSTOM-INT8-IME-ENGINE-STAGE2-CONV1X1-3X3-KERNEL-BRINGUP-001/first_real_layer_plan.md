# First Real Layer Plan

Stage 0 selected `/model.10/m/m.0/attn/MatMul` as the first real graph MMT4D target, but that activation path is asymmetric and needs zero-point correction.

Stage 2 brought up synthetic/local Conv1x1 and Conv3x3 kernels only. The Stage 0 graph inventory reports 102 `Conv` nodes in `manual_e2e_rep_conv_matmul_qdq.onnx`, but it does not list a concrete first Conv node. A direct ONNX inspection was attempted in Stage 2 and blocked because host Python lacks `onnx`:

```text
onnx_import_failed=ModuleNotFoundError("No module named 'onnx'")
```

Next stage should either install/use an already-approved converter environment with `onnx` or consume an existing graph metadata dump. It must select a real Conv node with:

- exact node name;
- input/output shapes;
- kernel/stride/pad/group;
- activation/weight scale and zero-point metadata;
- signedness and zero-point correction path;
- whether B weights can be prepacked once.

Recommended Stage 3 first real block target: earliest Conv or Conv+activation block whose weights are signed int8 with zero-point 0 and whose activation zero-point correction can be validated against the ONNX CPU oracle.
