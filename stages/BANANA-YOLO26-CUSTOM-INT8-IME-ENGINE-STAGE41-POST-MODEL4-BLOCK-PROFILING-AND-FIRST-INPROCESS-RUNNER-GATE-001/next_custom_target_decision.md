# Next Custom Target Decision

decision_status: provisional

Hard board selected-mode full-output gate did not pass, so Stage41 does not accept a new custom block target as final.

Best provisional target from exact host C++ in-process suffix profile:

```text
selected_next_block: model.16
reason: highest non-detect-head incremental cumulative delta with C2f-like operator mix and reusable model4-style kernels
incremental_delta_us: 16406.986
node_count: 66
conv_count: 9
operator_mix: Add:2, Concat:2, Conv:9, DequantizeLinear:17, Mul:9, QuantizeLinear:17, Sigmoid:9, Split:1
```

Rejected as first custom target:

```text
model.23: larger delta but high-risk detect/output head and postprocess-heavy.
model.22: includes attention/MatMul/Softmax and is higher risk.
model.5: immediate after model4, but lower ROI and high CV.
```

Required Stage42 first step:

```text
repair or explicitly scope the in-process ORT CPU reference contract on board before implementing model.16 custom kernels.
```
