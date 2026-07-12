# Persistent slice architecture

The executor uses one 1,638,400-byte arena, generated tensor and operation descriptors, lifetime-based reuse, one persistent worker pool, shared immutable packed weights, and resident `NCHWc8_SPATIAL_INNER_V1` storage. Prepare/run/destroy are explicit. Measured run performs no allocation, file I/O, Python, ORT call, internal NCHW/NHWC conversion, or float Q/DQ materialization.
