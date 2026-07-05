# Halo Overhead Report

The current Conv API supports symmetric `pad_h`; Stage18 uses local overcompute and row discard at spatial chunk boundaries.

| thread_count | overcomputed_rows | discarded_rows | estimated_extra_MACs | percent_overhead |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0.000000 |
| 2 | 2 | 2 | 737280 | 2.500000 |
| 3 | 2 | 2 | 737280 | 2.500000 |
| 4 | 2 | 2 | 737280 | 2.500000 |

For the target Conv:

```text
base MAC_count: 29491200
extra MAC_count with 2/3/4 threads: 737280
```

Future asymmetric padding support could remove this small overhead, but Stage18 did not need it to meet the strong acceptance gate.
