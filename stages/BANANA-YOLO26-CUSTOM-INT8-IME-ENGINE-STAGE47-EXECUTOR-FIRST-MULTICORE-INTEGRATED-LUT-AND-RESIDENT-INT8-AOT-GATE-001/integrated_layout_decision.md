
# Integrated layout decision

The diagnostic executor uses resident NHWC signed-int8 storage with explicit
ONNX uint8 zero-point metadata. There is one NCHW-u8 to NHWC-s8 entry adapter,
zero internal layout conversions, and one NHWC-s8 to NCHW-u8 exit adapter.
No float Q/DQ tensor is materialized in the measured slice.
