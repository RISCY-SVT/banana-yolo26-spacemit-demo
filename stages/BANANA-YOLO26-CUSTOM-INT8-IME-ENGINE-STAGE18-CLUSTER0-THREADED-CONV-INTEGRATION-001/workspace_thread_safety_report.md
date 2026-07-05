# Workspace Thread Safety Report

The Stage18 threaded Conv sidecar uses one independent worker state per cluster0 CPU.

Per worker:

```text
Y26PrepackedConvWeights: private
Y26ConvWorkspace: private
raw accumulator buffer: private
corrected accumulator buffer: private
output row range: disjoint
```

Shared state:

```text
input tensor: read-only
output tensor: disjoint row writes
barriers: synchronization only
current input/output pointers: set before barrier phase, read after barrier phase
```

Hot-loop allocation:

```text
worker threads: created at workspace creation
prepacked weights: created at workspace creation
workspaces: created at workspace creation
raw/corrected vectors: allocated at workspace creation
per-run heap allocation in y26_threaded_conv_run_ime_cluster0: none
```

Correctness evidence:

```text
thread_count 1/2/3/4: mismatches=0
checksum: 1324192976 for all thread counts
```
