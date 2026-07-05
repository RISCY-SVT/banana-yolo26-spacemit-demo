# Cluster0 Threading Feasibility Note

IME kernels remain cluster0 CPU0-3 only.

Stage16 does not enable multithreading by default and does not use OpenMP/all-core dispatch. CPU4-7 must not execute IME kernels.

No compact fixture threading headline evidence is accepted. Controlled threading feasibility should wait until representative/full-shape block/subgraph timing is stable.

Future candidate tests:

- 1 thread pinned to CPU0
- 2 threads pinned to CPU0-1
- 3 threads pinned to CPU0-2
- 4 threads pinned to CPU0-3
- CPU4 negative/protection check only if safe and separately authorized

Cluster1 may be explored later only for non-IME work and only with explicit scheduling/proof.
