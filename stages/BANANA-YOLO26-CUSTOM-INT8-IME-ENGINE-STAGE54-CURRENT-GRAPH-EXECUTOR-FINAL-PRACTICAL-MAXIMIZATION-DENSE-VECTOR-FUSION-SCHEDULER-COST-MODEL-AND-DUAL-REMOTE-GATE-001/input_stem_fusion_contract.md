# Input and stem contract

The selected explicit RVV quantizer preserves float32 RGB NCHW round-to-nearest-even, saturation, deterministic padded lanes, and state restoration. A compact C3 buffer is consumed directly by the dedicated stem. The fully fused float-to-stem arm was exact but substantially slower and is rejected.
