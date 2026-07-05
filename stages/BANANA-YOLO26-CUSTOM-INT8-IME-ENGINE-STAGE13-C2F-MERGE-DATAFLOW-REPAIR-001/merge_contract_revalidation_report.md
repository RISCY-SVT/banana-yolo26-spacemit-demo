# Merge Contract Revalidation Report

The accepted Stage 12 graph contract is preserved:

- `/model.2/m.0/Add` is float-domain.
- `/model.2/Concat` is float-domain.
- Concat input order remains:
  1. `/model.2/Split_output_0`
  2. `/model.2/Split_output_1_DequantizeLinear_Output`
  3. `/model.2/m.0/Add_output_0`
- post-Concat Q/DQ uses `scale=0.3288085460662842` and `zero_point_u8=2`.
- `/model.2/cv2/conv/Conv` consumes signed int8 storage equivalent to that post-Concat Q/DQ.

Stage 13 does not accept an integer-domain Add shortcut. `A2_fused_qdq_nhwc`
is still a float-domain merge candidate; it fuses buffer writes only.
