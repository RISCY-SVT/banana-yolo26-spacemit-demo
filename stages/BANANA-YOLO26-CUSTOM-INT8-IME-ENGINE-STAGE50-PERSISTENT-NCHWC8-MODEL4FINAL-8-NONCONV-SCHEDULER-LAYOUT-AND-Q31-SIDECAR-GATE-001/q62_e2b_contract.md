# Q62 E2b sidecar contract

E2b must preserve K1X_INT8_V1 Q62 assets, exact signed positive/negative RNE, saturation, LUT, and final store. Explicit RVV limb multiplication is implemented as a diagnostic, but a complete exact vector quotient/remainder/round/LUT path is not implemented.
