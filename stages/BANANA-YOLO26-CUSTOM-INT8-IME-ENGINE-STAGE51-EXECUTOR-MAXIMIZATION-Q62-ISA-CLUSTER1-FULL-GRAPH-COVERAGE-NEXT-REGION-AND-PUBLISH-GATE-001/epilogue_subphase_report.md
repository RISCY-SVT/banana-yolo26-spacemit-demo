# Q62 epilogue subphase attribution

E1 diagnostic worker sums show Q2 multiplier/RNE as the largest individual arithmetic bucket,
with bias, clamp, LUT, extraction, and store close enough that no single scalar instruction class
explains the full wall time. E2c removes the multiword Q62 path and writes contiguous C8 halves.
Instrumentation perturbs wall time; these worker sums are attribution only, not headline timing.
