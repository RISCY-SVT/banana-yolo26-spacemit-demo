# Thread Partition Boundary Report

Stage31 direct/sliding sidecar threading status:

- Primary direct sidecar implemented as single-thread CPU-local proof.
- No direct/sliding CPU0-3 threaded variant was promoted because the single-thread candidate failed both same-thread and best-threaded speed gates.
- Existing MMT4D 1-thread and 4-thread paths were measured in the same benchmark for comparison.

Affinity and CPU policy:

- Stage30 replay: CPU0, CPU1, CPU2, CPU3 only.
- Stage31 direct correctness: CPU0, CPU1, CPU2, CPU3 only.
- Stage31 stable benchmark command: `taskset -c 0-3`.
- No CPU4-7 IME execution was used.

Boundary/halo policy:

- Direct sidecar handles kernel padding locally with `input_storage_zero_point_s8`.
- No cross-thread halo partitioning was added in Stage31.

Conclusion:

Threaded direct/sliding work is deferred because the single-thread direct candidate is already much slower than current MMT4D.
