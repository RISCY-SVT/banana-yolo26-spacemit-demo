# Model16 Oracle Report

status: ready

The fixed host oracle package contains semantic and quantized cuts, deterministic input/output NPY files, exact raw hashes, graph metadata, scales/zero points, model/cut hashes, and the host ORT session contract. Full details are in `model16_oracle_manifest.tsv`.

Host generator contract:

```text
ORT: 1.27.0
provider: CPUExecutionProvider default
optimization: all
execution: sequential
intra/inter threads: 1/1
memory pattern/CPU arena/thread spinning: enabled
full model SHA256: 30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
```

Board ORT replay is diagnostic under Policy B:

| Contract | Mismatches | Max abs diff | Result |
|---|---:|---:|---|
| semantic float | 15590 / 409600 | 0.238198549 | board runtime differs |
| quantized uint8 | 15305 / 409600 | 11 | board runtime differs |

These differences do not invalidate the fixed host oracle. They confirm that board ORT cannot be used as the custom-kernel authority. `model16_oracle_replay.tsv` records complete aggregate statistics and hashes.

No scalar or optimized model16 implementation was introduced. Oracle readiness is not implementation authorization by itself.
