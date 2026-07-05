# RVV Merge Requant Report

Stage 13 did not introduce a new RVV merge kernel. It preserves the accepted
Stage 9/10 `A2_rvv_f32_lut` activation path and explicit RNE behavior.

Merge improvements are local dataflow changes:

- cached Split1 Q/DQ storage is reused and dequantized;
- Add and Concat temporary float buffers are avoided for A2;
- post-Concat QDQ writes signed int8 NHWC storage directly.

RNE regression tests passed on CPU0/1/2/3 through
`test_stage10_rvv_rounding_control`.
