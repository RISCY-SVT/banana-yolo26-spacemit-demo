# Zero-Point And Requant Boundary Report

Stage 2 raw kernels compute only:

```text
acc = sum(int8_a * int8_w)
```

They do not integrate activation zero-point correction, weight zero-point correction, requantization, output clamp, activation functions, or full graph scale propagation.

The local synthetic test `test_zero_point_correction_formula` verifies only the arithmetic identity:

```text
sum((a - za) * (w - zw))
= sum(a*w) - zw*sum(a) - za*sum(w) + K*za*zw
```

Status:

```text
zero_point_correction_integrated: synthetic-only
requant_integrated: false
full_graph_quantization_claim: false
```

Stage 3 may add row/column sum plumbing for a selected block, but it must compare against the ONNX CPU oracle before any model-level correctness claim.
