# Twenty-FPS feasibility

Target: pure model <=45 ms and full frame <=50 ms. The exact graph contains
`2.740154 GMAC`; meeting 45 ms requires `60.892 GMAC/s`
before any QDQ, activation, transpose, pooling, attention, head, or memory cost.
The best model5-geometry standalone diagnostic is M12xN16 at `54.360
GMAC/s`, which yields `50.407 ms` for MACs alone. Therefore
unchanged YOLO26n-640 cannot credibly satisfy the pure-model target.

The conservative current-graph AOT range is `105-130 ms`: 35 GMAC/s mixed-shape
delivery gives 78.29 ms of MAC work, with 27-52 ms reserved for aggressively fused
non-MAC/dataflow/head work. The current measured vendor ORT graph is 461.603 ms.

A tile-aligned K1X student at 416, 0.65-0.90 GMAC, simple static one-to-one head,
resident symmetric INT8, and global AOT scheduling projects to 24-38 ms. A 512
student projects to 35-50 ms and is the accuracy-first fallback. These are design
projections, not achieved latency. Training and accuracy are unproven.
