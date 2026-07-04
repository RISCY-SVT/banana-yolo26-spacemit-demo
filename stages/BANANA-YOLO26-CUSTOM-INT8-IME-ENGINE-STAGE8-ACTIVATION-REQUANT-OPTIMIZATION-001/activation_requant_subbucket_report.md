# Activation Requant Subbucket Report

activation_mode: `scalar_float_reference`
scope: diagnostic profiling only

## Act0

| subbucket | us |
|---|---:|
| corr_i32_to_conv_out_quant_code | 129584 |
| conv_out_code_to_float_dequant | 2675.05 |
| float_silu_sigmoid_mul | 86571.4 |
| act_quant_float_to_uint8 | 101925 |
| signed_storage_shift | 6310.51 |
| layout_or_pack_handoff | 0 |
| combined_current_fallback | 327070 |

## Act1

| subbucket | us |
|---|---:|
| corr_i32_to_conv_out_quant_code | 64440 |
| conv_out_code_to_float_dequant | 1336.05 |
| float_silu_sigmoid_mul | 43175.8 |
| act_quant_float_to_uint8 | 51179 |
| signed_storage_shift | 4849.88 |
| layout_or_pack_handoff | 0 |
| combined_current_fallback | 164984 |

## Interpretation

The old fallback is not only `expf` bound. Conv-output requant and activation-output quantization are also major buckets. The LUT path removes `conv_out_code_to_float_dequant`, `float_silu_sigmoid_mul`, and `act_quant_float_to_uint8` from the hot path, but still pays per-element conv-output code quantization.
