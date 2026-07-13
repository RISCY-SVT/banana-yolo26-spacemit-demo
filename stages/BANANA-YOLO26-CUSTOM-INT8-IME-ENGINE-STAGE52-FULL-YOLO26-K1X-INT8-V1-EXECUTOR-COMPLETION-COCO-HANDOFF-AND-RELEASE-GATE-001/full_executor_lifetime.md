# Arena lifetime

`full_executor_arena.tsv` and the package tensor manifest give every offset,
size, first producer, and last consumer. Offsets are generated offline with
64-byte alignment and lifetime-based reuse. Prepare allocates the arena,
worker scratch, optimized-core live-out snapshots, and optional diagnostic
boundary buffers. The run path only reuses those buffers.
