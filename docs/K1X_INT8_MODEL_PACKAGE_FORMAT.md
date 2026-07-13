# K1X INT8 Model Package Format

The package profile is
`K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001`, schema version 2, little-endian. The
root `asset_hashes.tsv` is trusted by SHA-256 at prepare time. Hashes provide
reproducibility and accidental-corruption detection, not signing or
authenticity.

Key files:

```text
package.json            identity, model lineage, graph dimensions
tensors.tsv             numeric tensor IDs, shapes, layouts, arena offsets
operations.tsv          numeric static operation schedule and assets
head_assets.tsv         fixed box and class lookup assets
assets/<op>/...         integer weights, packed weights, sums, bias, Q62/M63,
                        shifts, LUTs, and fixed Softmax tables
optimized_core/...      verified resident model4-final through model8 package
asset_hashes.tsv        size and SHA-256 for every package file
```

Dense Conv assets include OIHW weights, N16/K8 packed weights, weight sums,
bias_i32, corrected bias_i64, Q62 multipliers, right shifts, and M63 values.
Prepare recomputes packing, sums, corrected bias, and M63 invariants from the
loaded bytes and rejects any mismatch.

The package generator derives arithmetic offline from exact float32 bit
patterns. Board prepare performs no floating-point scale derivation. The hot
path performs no package parsing or validation.
