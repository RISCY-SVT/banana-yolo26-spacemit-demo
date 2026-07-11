# Custom model5 worker scaling

The unchanged exact Stage43 R0 route was measured on CPU0-3 with 1/2/3/4 persistent workers, warmup 10, runs 100, repeats 5. Every arm was exact for F0, had stable output hash, reported `affinity_ok=1`, and had no SIGILL.

Three workers were fastest: `23516.5 us`. Four workers regressed to `25265.3 us`, with higher process CPU time and synchronization/imbalance. The selected bounded candidate therefore uses three workers. These continuity results predate R2a and are not substituted for the final paired R0/R2a measurement.

Raw evidence: `run_logs/0037_custom_model5_worker_stable.stdout`.
