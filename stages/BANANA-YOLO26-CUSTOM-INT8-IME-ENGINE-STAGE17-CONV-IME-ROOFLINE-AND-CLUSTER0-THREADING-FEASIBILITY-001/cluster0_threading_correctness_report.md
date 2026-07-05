# Cluster0 Threading Correctness Report

Threading candidate: `spatial_row_split`

The selected representative Conv work was partitioned over output rows. Top/bottom row chunks use local overcompute plus discarded halo rows because the current Conv params support symmetric `pad_h`, not asymmetric per-side padding.

All accepted threading paths:

```text
CPU set: CPU0-3 only
IME on CPU4-7: not executed
mismatches: 0
checksum: 1324192976
```

| threads | CPUs | mismatches | checksum |
|---:|---|---:|---:|
| 1 | `0` | 0 | 1324192976 |
| 2 | `0-1` | 0 | 1324192976 |
| 3 | `0-2` | 0 | 1324192976 |
| 4 | `0-3` | 0 | 1324192976 |
