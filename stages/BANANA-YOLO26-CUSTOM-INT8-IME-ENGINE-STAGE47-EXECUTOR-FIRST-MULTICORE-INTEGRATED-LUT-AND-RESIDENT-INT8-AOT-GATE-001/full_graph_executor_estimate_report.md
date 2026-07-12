
# Full-graph executor estimate

Mapped full-shape integrated measurements cover `97.268007%` of graph
MACs. All materially profiled non-MAC classes are mapped to either measured
resident-int8 rows or explicit conservative B120 fallbacks.

- Optimistic: `225.228387 ms` (every MAC at the best measured integrated rate; deliberately unattainable upper bound).
- Central: `1168.164550 ms` (nearest class/shape mapping).
- Conservative: `1277.559421 ms` (p95 mapping plus fallback allowance).

Even the optimistic full-work bound exceeds 80 ms and does not depend on ideal
four-core scaling. The estimate is decision evidence, not measured full-model
latency and not a model FPS claim.
