# Complete model23 head

All three one-to-one detection scales are executed inside the static executor. The graph path includes the regression and classification Conv branches, grouped Conv nodes, quantized transformations, and final deterministic 300-row selection. No external NMS or hidden Python postprocessor is part of the pure-model timing.

The frozen output schema is `[x1, y1, x2, y2, score, class]` with 300 rows. Boxes use package Q16 assets and scores use package Q24 assets before exact float serialization. Point ranking is score descending then point index ascending. Final candidate ranking is score descending, point slot ascending, then class index ascending. N4/N8 portable direct references exist, while the selected IME functional route currently masks unused lanes of N16; this is reported as an optimization limitation, not hidden work.
