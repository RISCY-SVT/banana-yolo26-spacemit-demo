# Stage31 Board CPU0-3 Report

Board:

- Hostname: `bf3`
- Kernel: `Linux 6.6.63`
- CPU: SpacemiT X60
- CPU0 governor: `performance`
- CPU0 current frequency during anchor: `1600000`

Deployed binaries:

| Binary | SHA256 |
| --- | --- |
| `bench_stage30_vmadot123_probe` | `e2069af7208af3fd9b2fdf21f5692fae7a3822bee294c9ea168e6feeaa7e50ca` |
| `bench_stage31_vmadot123_direct_conv` | `c4f4f8634cc6f2eeeafc3abf6dddcc7cfb94f0f16724fda29ecb6b52f6aa2a8b` |

CPU0-3 coverage:

- Stage30 `vmadot1/2/3` replay passed on CPU0, CPU1, CPU2, CPU3.
- Stage31 direct Conv correctness passed on CPU0, CPU1, CPU2, CPU3.
- No CPU4-7 IME execution was used.

Stable timing:

`taskset -c 0-3`, `warmup=10`, `runs=100`, `repeats=5`.
