# Predeclared hybrid-boundary decision rules

These rules were fixed before any Stage65B-R1 scout, boundary, hybrid, or
full-COCO result was available.

The global candidate is selected by highest 500-image scout mAP50-95, then
highest scout AP-large, then lexicographic lane ID. All required lanes must
first pass generation reproducibility, graph conformance, and host semantics.

For metric `M`, define its recoverable gap and an arm's recovery fraction as:

```text
gap(M) = H8(M) - H0(M)
recovery(arm, M) = (arm(M) - H0(M)) / gap(M)
```

Fractions are evaluated for mAP50-95 and AP-large. A gap with absolute value
below `0.001` is treated as too small to localize.

Decision order:

1. `p5-confidence-causality-supported` when H1 recovers at least 50% of both
   gaps and H6 improves AP-large recovery over H1 by less than 20 percentage
   points. H5 must not improve AP-large recovery over H1 by 20 points or more;
   otherwise the P5 bbox branch is also material.
2. `multi-scale-confidence-causality-supported` when H6 recovers at least 50%
   of both gaps and improves AP-large recovery over H1 by at least 20 points.
3. `pyramid-boundary-hypothesis-not-supported` when H6 recovers less than 25%
   of both gaps while H8 has a measurable recoverable gap.
4. `earlier-subgraph-or-tail-interaction` for remaining cases, including a
   materially non-FP32 H8 control or mixed recovery that the named boundary
   hypotheses do not isolate.

Graphwise, clipping, cosine, and H500 results are supporting diagnostics only.
They cannot select a causal classification without the full-COCO hybrid arms.
