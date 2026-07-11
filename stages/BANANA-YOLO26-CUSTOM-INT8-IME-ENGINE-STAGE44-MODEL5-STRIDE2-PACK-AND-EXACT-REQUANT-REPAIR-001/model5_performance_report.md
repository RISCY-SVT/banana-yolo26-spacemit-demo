# Model5 performance

Performance authority is the paired instrumentation-off measurement. With three CPU0-3 workers, R0 is `24636.0 us` and R2a is `24157.4 us`; paired delta is `-478.655 +/- 12.4319 us`, or `-1.94291%`.

The resource-matched isolated board ORT selection baseline is intra4 at `11701.121842 +/- 31.218416 us`. R2a is `106.4537%` slower. It also loses to the intra1 continuity arm (`18169.770948 us`). Therefore model5 compute status is negative despite a small local R2a improvement.

No model FPS or production latency claim is made. Full-board ORT timing is a strategy diagnostic only.
