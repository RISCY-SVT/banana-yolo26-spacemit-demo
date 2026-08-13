# Complete frontier proof

Frontiers were frozen before reading new task metrics. For every accepted source partition, the cut is the exact set of all node outputs produced upstream and consumed downstream. Initializers remain immutable graph constants; graph inputs may not cross a frontier. Static shape/dtype, unique names, unique source-producer/output-index mapping, and direct source-op -> Q -> DQ provenance are mandatory. Extracted source halves were checked for partition leakage.

- C0: 1 live tensors, 22 upstream nodes, status pass.
- C1: 2 live tensors, 77 upstream nodes, status pass.
- C2: 3 live tensors, 113 upstream nodes, status pass.
- C3: 3 live tensors, 155 upstream nodes, status pass.
- C4: 3 live tensors, 190 upstream nodes, status pass.
- C5: 4 live tensors, 264 upstream nodes, status pass.
- C6: 3 live tensors, 304 upstream nodes, status pass.
- C7: 6 live tensors, 358 upstream nodes, status pass.
