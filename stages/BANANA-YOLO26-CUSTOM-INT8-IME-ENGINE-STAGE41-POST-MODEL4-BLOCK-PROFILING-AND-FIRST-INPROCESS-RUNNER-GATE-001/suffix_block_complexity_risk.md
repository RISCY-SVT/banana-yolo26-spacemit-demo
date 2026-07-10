# Suffix Block Complexity And Risk

Risk summary:

```text
model.5: low structural complexity, immediate after model4, but lower ROI than model16 and high timing CV.
model.6: medium, C2f-like, clean reusable pattern, moderate ROI.
model.16: medium, C2f-like, clean reusable pattern, highest non-head ROI.
model.22: high, includes attention/MatMul/Softmax and quantized Conv mix.
model.23: high, detect/output head and postprocess-heavy operators.
```

The first expansion should not start with `model.23` despite high cumulative delta, because it mixes detection-head postprocess with many Conv/QDQ nodes and would turn Stage42 into a broad runtime/output-surface task.

`model.16` is the best provisional target once board in-process ORT contract is repaired or explicitly scoped.
