# Constrained range-policy API

Development version: `2.1.2+riscy.2.dev1`.

An entry in `quantization_parameters.custom_setting` selects either explicit
`tensor_names` or a bounded `input_names`/`output_names` region. The nested
`range_policy` supports `enabled`, `strict`, `lock_qparams`, `objective`,
`preserve_zero`, `required_real_min`, `required_real_max`,
`semantic_floor`, `percentile`, `search_steps`, and `scale_epsilon`.

Supported objectives are `minmax`, `percentile`, `mse`, `kl`, and
`constrained-mse`. Selection enumerates legal signed-INT8 zero points,
simulates ONNX round-half-even and saturation, and uses deterministic tie
handling. The SiLU floor is opt-in and never global.

The feature changes activation scale/zero point only. It does not remove Q/DQ,
merge output branches, alter output names/order/shapes, or quantize the tail.
