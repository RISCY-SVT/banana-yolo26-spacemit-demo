# Suffix Block Profile Report

Suffix profiling used C++ in-process ORT C API on the host exact scaffold because the board selected-mode ORT CPU contract did not pass byte-exactness.

The profile is cumulative from `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output` to each candidate boundary. Incremental deltas are approximate block costs.

Top incremental deltas:

```text
model.23: 29408.924 us, high risk detect/output head
model.16: 16406.986 us, Conv/C2f-like, 66 nodes, 9 Conv, reusable model4-style kernels
model.5: 14345.387 us, immediate Conv/activation block, smaller but high CV
model.6: 10160.558 us, Conv/C2f-like, 66 nodes, 9 Conv
model.22: 10109.558 us, high risk attention/MatMul/C2f-like mix
model.13: 9249.042 us, Conv/C2f-like
```

`model.23` is the largest cumulative delta, but it contains the detect/output head and postprocess-heavy operators. It is not the safest first custom expansion target.

Provisional next custom target from exact host profile: `model.16`.

This is provisional because board selected-mode full-output correctness is blocked.
