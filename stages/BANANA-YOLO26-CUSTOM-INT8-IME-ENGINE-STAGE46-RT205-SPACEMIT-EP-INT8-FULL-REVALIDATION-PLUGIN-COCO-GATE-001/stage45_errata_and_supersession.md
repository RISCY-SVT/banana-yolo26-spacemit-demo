# Stage45 errata and supersession

Stage45 history and raw evidence remain unchanged. Stage46 records these narrower
interpretations:

1. `50.407 ms` is a CPU0-only prepacked M12xN16 compute projection, not a
   physical four-core graph lower bound.
2. `105-130 ms`, `24-38 ms`, and `35-50 ms` are analytical envelopes, not
   measured graph latencies.
3. Scalar read/transpose/reference requant/hash-assisted pack rows are reference
   diagnostics, not optimized production primitives.
4. M12 full-shape tails and a fused epilogue were not measured.
5. Stage45 FP32 accuracy used only ORT_ENABLE_ALL. Stage46 supplies the full
   FP32/INT8 disable/all host matrix.
6. The Stage45 repository report has a pending end-head marker; the actual head
   is recorded in `stage45_traceability_addendum.md`.
7. Candidate latency and strategy tables are non-empty analytical artifacts.
8. Both 416 latency-first and 512 accuracy-fallback student hypotheses remain
   untrained and unselected.

The reported duplicate operator line in `model_executor_codesign_spec.md` was
not present at Stage46 preflight, so no historical source was rewritten.
