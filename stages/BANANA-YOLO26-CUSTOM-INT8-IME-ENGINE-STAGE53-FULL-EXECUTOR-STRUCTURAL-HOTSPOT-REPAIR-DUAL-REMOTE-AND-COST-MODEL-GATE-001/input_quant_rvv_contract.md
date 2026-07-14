# Input quantization RVV contract

The explicit RVV input path converts float32 RGB NCHW in [0,1] directly to signed NCHWc8 storage using exact round-to-nearest-even, saturation, deterministic padded lanes, and restored floating/vector state. It materializes no intermediate uint8 tensor.
