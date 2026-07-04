# Rounding Mode Regression Report

scope: Stage 12 preflight and board validation

## Existing RVV RNE Regression

Board CPU0/1/2/3 ran:

```text
test_stage10_rvv_rounding_control
```

For both compact fixtures and ambient `frm=0..4`:

- `mismatches=0`
- `after_frm` equals the incoming ambient `frm`

## Stage 12 New Boundary

The new Add/Concat path is scalar float fallback plus post-Concat QDQ. The
post-Concat QuantizeLinear oracle matched ONNX Runtime on compact fixtures:

- `concat_q_mismatches_against_ort=0`
- `concat_q_max_abs_diff_u8=0`
