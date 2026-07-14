# Exact E2c3 C8 contract

E2c3 processes eight channels as two explicit four-lane e64 vsmul groups, narrows semantic indices, performs explicit indexed byte LUT loads, and writes one contiguous signed-storage C8 group without a scalar per-lane LUT loop or stack round-trip.

It remains K1X_INT8_V1 Q62/RNE and restores vcsr/vxrm/vxsat.
