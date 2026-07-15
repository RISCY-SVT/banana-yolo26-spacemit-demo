# Fused dense/E2c4 contract

The bounded M8xN16 symbol consumes accumulator groups destructively and performs exact corrected bias, M63 vsmul/RNE, zero point, clamp, narrow, indexed activation LUT, and C8 store without a C-tile scratch round trip. It preserves K1X_INT8_V1 and vector CSR state.
