
# AOT executor architecture

The substrate has generated tensor and operation TSVs, one 1,638,400-byte arena,
compile-time-like fixed offsets loaded only during prepare, lifetime reuse,
immutable prepared weights, one persistent CPU0-3 worker pool, explicit quant
metadata, and separate prepare/run/destroy/diagnostic surfaces. The measured run
has no graph-name lookup, registry dispatch, allocation, file I/O, Python, or ORT.
