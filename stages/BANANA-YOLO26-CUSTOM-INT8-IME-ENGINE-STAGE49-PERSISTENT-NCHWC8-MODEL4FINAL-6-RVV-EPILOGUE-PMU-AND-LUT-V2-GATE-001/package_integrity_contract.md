# Package integrity contract

The loader accepts `K1X_INT8_V1`, schema 2, profile `K1X_INT8_V1_GENERAL`, layout `NCHWc8_SPATIAL_INNER_V1`, and little-endian assets only. The runner supplies trusted manifest SHA-256 `0d3c3d49abdc8dd83857af223ea63bcb7a31058be4bcdb7cd7e6ccdf35659bac` externally.

Prepare verifies required files, exact sizes and hashes, model/source lineage, no symlink or executable entries, and no unexpected files. It then recomputes packed-weight sums and bias-inclusive accumulator bounds from loaded bytes and rejects metadata disagreement. Integrity is proven; cryptographic authenticity/signing is not claimed. Conv and slice APIs reject overlapping input/output ranges.
