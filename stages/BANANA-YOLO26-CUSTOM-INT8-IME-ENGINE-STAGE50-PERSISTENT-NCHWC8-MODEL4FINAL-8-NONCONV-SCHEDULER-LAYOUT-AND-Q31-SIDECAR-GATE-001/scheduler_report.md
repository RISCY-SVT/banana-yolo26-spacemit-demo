# Scheduler result

Active-worker-only completion is selected. CPU0-3 remain IME workers and CPU4 is controller/non-IME only. The stage did not require boot-time isolation, CPU4-7 IME, or an unbounded spin pool. Stable full-slice p95, not the scout alone, is the final tail-latency evidence.
