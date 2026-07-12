
# Fused epilogue contract

The integrated route applies zero-point correction, int32 bias, exact per-channel
Q62 multiplier/shift with integer round-to-nearest-even, saturation, optional
256-entry activation LUT, and the final resident signed-code store in the output
tile. It allocates no global corrected-int32 tensor and uses no float arithmetic
in the hot epilogue.
