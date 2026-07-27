# Performance raw evidence

The bounded 500-sample rows retained in Git are in
`../performance_raw_500.tsv` (SHA-256
`377b8c0bda696d597da5bbcb99a47e26e0c7e1de3eb02b6c412215a50556afbb`).

The complete 1,000-sample stability table is retained under the Stage63 shared
NVMe log root at:

```text
artifacts/performance/stability/stability_samples.tsv
```

Its SHA-256 is
`4259f957999b080d913ede5d6c4f784477ca692030ee13884e7d7960ac9f182e`.
Per-arm stdout, `/usr/bin/time -v` output, commands, output tensors, and
before/after frequency and temperature snapshots are adjacent to that file.
Large raw evidence is intentionally not committed.
