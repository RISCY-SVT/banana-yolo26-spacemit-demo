# Depthwise RVV contract

All eight graph grouped convolutions are true 3x3 depthwise forms with group=input_c=output_c and channel counts divisible by eight.

The selected explicit RVV path processes one C8 block, separates branch-free interior from exact padded borders, accumulates widened products in int32 lanes, applies exact Q62 E2c2 requantization, and writes one contiguous C8 result.
