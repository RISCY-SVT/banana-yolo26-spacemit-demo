# ORT model5 thread fairness

The semantic oracle remains host ORT 1.27.0 with `ORT_DISABLE_ALL`, sequential, intra/inter 1. This report is performance-only board evidence from vendor ORT 1.20.2+spacemit, `ORT_ENABLE_ALL`, sequential, inter 1, CPU0-3.

The same isolated model5 cut and identical F0 bytes were measured with warmup 10, runs 100, repeats 5. Intra 4 is the fastest stable resource-matched arm at `11701.121842 +/- 31.218416 us`. Intra 1 is the continuity arm at `18169.770948 +/- 25.129083 us`; intra 2 is `13137.551702 +/- 6.125398 us`.

All arms produced the same board-ORT output hash. This does not make board ORT the correctness authority. It recalibrates the performance gate and materially raises the bar relative to Stage43's one-thread comparison.

Raw evidence: `run_logs/0035_ort_model5_thread_stable.stdout` under the Stage44 log root.
