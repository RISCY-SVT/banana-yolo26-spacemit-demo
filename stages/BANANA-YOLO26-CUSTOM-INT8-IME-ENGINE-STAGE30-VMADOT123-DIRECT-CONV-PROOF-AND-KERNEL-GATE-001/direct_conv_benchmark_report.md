# Direct Conv Benchmark Report

No direct Conv sidecar benchmark was accepted in Stage30.

Reason:

- Stage30 reached the instruction semantics proof gate.
- The first valid Conv candidate requires an expanded-A-panel schedule and non-overlapping output-row handling.
- That is a Stage31 integration/benchmark task, not a proof-lane microprobe.

Benchmark status:

| candidate | status | result |
| --- | --- | --- |
| direct `vmadot1/2/3` 3x3 Conv sidecar | not accepted in Stage30 | no speed win measured |
| current Stage28 MMT4D threaded path | reference only | unchanged |
