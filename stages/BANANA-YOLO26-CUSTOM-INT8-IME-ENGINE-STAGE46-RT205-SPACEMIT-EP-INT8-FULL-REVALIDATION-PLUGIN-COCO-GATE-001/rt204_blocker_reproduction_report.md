# RT204 blocker reproduction

The matched Stage46 RT204 runner reproduced the accepted historical control.
CPU output for `15_conv_qdq_attr_kernel_shape.onnx` is byte-identical to the old
probe. SpacemiT EP fails at `synthetic/conv/Conv_token_1` with
`output_type not implemented for clip minmax`. The real YOLO26 first-Conv cut
fails on the same compiler path. The RT205 comparison is therefore valid.
