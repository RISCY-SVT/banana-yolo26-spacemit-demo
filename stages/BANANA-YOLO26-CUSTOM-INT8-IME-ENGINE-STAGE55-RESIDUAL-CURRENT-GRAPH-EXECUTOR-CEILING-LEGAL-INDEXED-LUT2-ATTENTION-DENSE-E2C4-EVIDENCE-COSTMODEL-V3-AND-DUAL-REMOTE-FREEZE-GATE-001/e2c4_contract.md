# Exact E2c4 contract

E2c4 loads two raw C4 int32 accumulator groups, vector sign-extends to C8 int64, adds corrected bias, applies exact M63 vsmul/RNE, zero point, clamp, narrow, optional proven LUT, and direct contiguous C8 store. It removes the per-row corrected[8] stack roundtrip and restores vcsr/vxrm/vxsat.
