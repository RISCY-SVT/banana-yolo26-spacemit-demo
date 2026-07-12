# Persistent slice lifetime

Offsets and first/last operation lifetimes are in `persistent_slice_arena.tsv`. Inputs and outputs may not overlap. Split outputs are views represented by descriptor offsets; Concat writes directly to package-defined channel destinations. Reuse occurs only after a tensor's recorded last consumer.
