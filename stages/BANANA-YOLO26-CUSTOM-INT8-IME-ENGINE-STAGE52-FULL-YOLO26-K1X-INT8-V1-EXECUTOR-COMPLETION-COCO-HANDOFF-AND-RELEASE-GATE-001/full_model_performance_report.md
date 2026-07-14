# Full-model performance

The headline surface is preprocessed `1x3x640x640` float32 input through the static integer schedule and fixed `1x300x6` output. It excludes package loading and file I/O.

An earlier queued wrapper omitted the OpenCV runtime directory. Its custom executable arms failed in the dynamic loader before executor creation, produced identical error-log hashes, and are retained only as failed raw evidence. Every value in this report comes from the corrected arm, which passed explicit loader and version preflight.

The first preloaded-image attempt overlapped a stale Stage 52 process on the same IME cores. That partial log is retained as `preloaded_pipeline.excluded-overlap.log` and is excluded from every summary. The reported preloaded pipeline was rerun with no other Stage 52 executor process present.

SCHED_OTHER mean was 504137.644000 us, p95 506182.200000 us, and p99 527434.100000 us (1.983585 pure-model FPS). SCHED_RR priority 20 was measured only as a lab sidecar: mean 503473.764000 us and p95 515912.800000 us. SCHED_OTHER remains the handoff default.

The 10,000-run SCHED_OTHER soak produced p99 527271.240000 us, p99.9 546540.109000 us, and maximum 595021.000000 us. Output hashes and CPU affinity were stable, and CPU4-7 IME count remained zero.

The matched B120 ORT repeat-mean surface was 460208.112134 us. The custom mean delta was +9.545580% and the ORT/custom ratio was 0.912862x.

The ORT distribution contains five repeat means, while custom headline percentiles use 500 per-inference samples. `full_model_ort_comparison.tsv` labels this statistical-unit difference; no cross-unit percentile claim is made.
