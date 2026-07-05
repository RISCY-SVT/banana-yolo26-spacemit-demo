# Merge Oracle Report

Oracle source: existing Stage 12 compact fixtures generated from the ONNX CPU
micro-oracle for Add + Concat + post-Concat Q/DQ and `/model.2/cv2/conv/Conv`.

## CPU0-3 Board Correctness

`test_stage13_merge_dataflow` passed on CPU0, CPU1, CPU2, and CPU3.

Checked outputs:

- `concat_mismatches=0`
- `model2_cv2_mismatches=0`

Checked candidates:

- `A0_scalar_float_merge`
- `A1_fused_add_concat`
- `A2_fused_qdq_nhwc`
- `A2_rvv_f32_lut_fused_qdq_nhwc`
- `A2_ime_fused_qdq_nhwc`

No CPU4-7 IME execution was performed.
