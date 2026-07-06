# Workspace Thread Safety Report

Design:

```text
one persistent threaded Conv workspace per prepared runner
one worker context per CPU0-3 worker
non-overlapping output row ranges
activation sidecar uses the same worker pool and phase barrier
no shared mutable output region between workers
zero-row workers no-op with success
```

Correctness evidence:

```text
Stage19 thread counts 1/2/3/4: mismatches=0
Stage19 activation sidecar: mismatches=0
checksum stable: -143848
Stage18 representative replay checksum stable: 1324192976
```

Known limitation:

```text
The Stage19 C2f fixture is compact.
Representative/full-shape model4 C2f workspace pressure is not yet proven.
```
