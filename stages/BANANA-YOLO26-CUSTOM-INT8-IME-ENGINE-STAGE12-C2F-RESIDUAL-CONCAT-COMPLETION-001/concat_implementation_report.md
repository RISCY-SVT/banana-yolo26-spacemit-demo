# Concat Implementation Report

implementation: `float materialize + post-Concat Q/DQ`

## Oracle

`custom_int8_engine/tools/extract_stage12_c2f_oracle.py` builds a compact ONNX
micro-oracle with:

```text
Add -> Concat(axis=1) -> QuantizeLinear
```

Result:

- `concat_q_mismatches_against_ort=0`
- `concat_q_max_abs_diff_u8=0`

## Timing

CPU0 full-shape Stage 12 IME A2:

- `concat_us=4335.56`
- `post_concat_qdq_us=83007.7`
- `add_concat_share_pct=15.4979`

The QDQ loop is the larger part of the merge bucket.
