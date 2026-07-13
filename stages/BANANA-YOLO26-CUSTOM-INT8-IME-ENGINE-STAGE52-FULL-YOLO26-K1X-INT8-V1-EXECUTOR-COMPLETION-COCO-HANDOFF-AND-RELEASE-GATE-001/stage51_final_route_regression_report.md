# Stage 51 final-route regression

Stage 52 rebuilt and redeployed the exact Stage 51 final commit before full-graph work. The selected E2c route remained exact and stayed inside the predeclared ceilings.

| Surface | Scheduler | Mean (us) | p95 (us) | Status |
|---|---|---:|---:|---|
| model5 E2c | SCHED_OTHER | 3554.263464 | 3623.952200 | pass |
| model4-final to model9 | SCHED_OTHER | 21233.575084 | 21371.511150 | pass |
| model4-final to model9 | SCHED_RR 20 | 21035.608008 | 21177.959250 | pass |

The 10,000-run SCHED_OTHER soak measured mean 21258.538130 us, p95 21485.160000 us, p99 24155.184000 us, p99.9 26770.333100 us, and max 28850.400000 us. The matching RR20 soak measured mean 20986.041060 us, p95 21102.000000 us, p99 23806.624000 us, p99.9 26336.016100 us, and max 26717.400000 us. Output hashes were stable, vector state was restored, and no CPU4-7 IME execution occurred.
