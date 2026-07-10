# Execution Provider Assignment Evidence

Provider registration alone is not node assignment evidence.

For the board `ORT_DISABLE_ALL` model.0 cut, verbose vendor ORT logs state:

```text
Adding default CPU execution provider.
All nodes placed on [CPUExecutionProvider]. Number of nodes: 6
```

The same log records every executed node, including the first Conv and QuantizeLinear, as `Placement:'CPUExecutionProvider'`. ORT profiling was enabled and produced `stage42_profile_ort_only_2026-07-10_08-59-11.json`, SHA-256 `12e9cdef9e270d0c03316772fb817ba2e81b9176bdd518c7ad93362d181baae5`.

Classification:

- provider registered: proven.
- provider appended: no; default CPU provider was used.
- node assignment: observed for the six-node model.0 diagnostic cut.
- full-graph node assignment: not separately observed.
- fallback on probed cut: none observed.

This evidence is sufficient to attribute the first divergence probe to CPUExecutionProvider execution, but it is not generalized to every possible board session.
