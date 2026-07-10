# Layout Policy

Current scaffold layout crossings:

```text
ORT fallback: NCHW
custom model4 runner: NHWC at quantized boundary
```

Policy:

```text
1. Time every NCHW/NHWC adapter.
2. Do not hide adapter time in Conv buckets.
3. Prefer keeping quantized tensors resident in the layout consumed by the next custom block.
4. Avoid float round-trips at Q/DQ boundaries unless they are part of the accepted ONNX oracle surface.
5. Do not change ONNX semantics for speed.
```
