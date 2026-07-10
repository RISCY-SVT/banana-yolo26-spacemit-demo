# Runner Timing Repair Report

## Repair

The runner now has explicit `validate`, `benchmark`, `profile`, `same-input-model4`, and `ort-only` modes. `benchmark` constructs only prefix and suffix ORT sessions plus the custom model4 runner. It does not create or execute the full ORT reference or ORT model4 reference in warmup or measured iterations.

The benchmark output explicitly records:

```text
reference_in_hot_loop=0
model4_ort_session_created=0
file_io_in_hot_loop=0
local_layout_allocation_in_hot_loop=0
ort_output_allocation=runtime_managed_unavoidable
```

Layout buffers are allocated once in `Model4Runner` and reused. Comparisons and dumps happen after timing. ORT-owned output allocation remains unavoidable and is reported rather than hidden.

## Corrected board timing

Protocol: CPU0-3, warmup 10, runs 100, repeats 5, performance governor at 1.6 GHz.

```text
custom pipeline mean: 826008.582826 us
custom pipeline stddev: 1095.337938 us
CV: 0.132606%
prefix: 229662.287042 us
layout in: 5293.200918 us
custom model4: 25149.098496 us
layout out: 11838.504756 us
suffix: 554052.102166 us
attribution: 99.998379%
```

Separate stable arms:

- full board ORT: `796866.390970 +/- 653.341854 us`.
- board ORT model4 same input: `37072.764532 +/- 27.809221 us`, numerically non-authoritative.
- board custom IME same input: `26815.626798 +/- 121.702623 us`, byte exact.
- board custom scalar same input: `378322.274994 +/- 165.597470 us`, byte exact.

Stage41's `858404.224484 us` total included one model4 ORT reference run in every iteration and omitted it from attribution. The Stage42 measurement structurally removes that work; absolute subtraction across sessions is not treated as an exact speedup.

The pipeline final output remains diagnostic because its prefix/suffix use board ORT. Versus fixed host output0 it has 1604 mismatches and max diff `635.420248985`; versus board full ORT it has 1508 mismatches and max diff `635.707154751`. The model4 same-input correctness gate is independent and passes exactly.

These are scaffold component timings, not model FPS or production latency. Raw repeats are in `pipeline_timing_raw.tsv`; aggregates are in `pipeline_timing_summary.tsv`.
