# Epilogue report

E1 removes per-element asset validation and correction multiplication by validating once at prepare, precomputing corrected bias, inlining exact Q62 RNE, and storing contiguous C8 output. In the matched scout E1 reduced M12/P3 from 8248.018550 us to 4824.669833 us and remained exact.

E2 was explicitly blocked. GCC could not vectorize the exact signed 64x64-to-128 Q62 loop and emitted scalar `mul/mulh`; the available RVV surface did not provide a proven exact replacement. No Q31 or floating approximation was allowed.
