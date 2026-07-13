# Detection head contract

All three scales, dense and grouped Conv rows, N4/N8/N16 tails, reshape/split/
transpose surfaces, TopK, Gather-equivalent selection, and final
`1x300x6` output are internal to the executor.

Head selection first chooses the 300 highest per-point class scores, then the
300 highest point/class candidates. Ties use score descending, point slot
ascending, and class ascending. Box values use package Q16 tables; class scores
use package Q24 sigmoid tables. Output order is deterministic.
