# Full executor architecture

Prepare verifies the deterministic package and builds:

- one 8,192,000-byte activation arena with lifetime reuse;
- immutable raw and offline-packed weights;
- per-worker bounded A-panel and accumulator scratch;
- a prepare-bound operation function table;
- a persistent four-worker pool on CPU0-3;
- a controller pinned to CPU4.

The safe default is `SCHED_OTHER`. IME is confined to the worker pool.
The measured run has no ORT, Python, per-run allocation, per-run file I/O,
runtime package parse, float Q/DQ materialization, or per-operator layout
adapter. Diagnostic boundary capture is prepare-time selected and excluded
from headline timing.
