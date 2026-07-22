# R768 Q0 Graph Audit

The R768 graph is a deterministic static-shape rewrite of the accepted source
model. The source model SHA-256 is
`30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`.

- Static model SHA-256: `3bb1695a5506b9e0c15ce4c511c30d3006db212c7c0c4ff5fb2c289183edfc8b`
- Profile: `K1X_INT8_V1_YOLO26N_768_FULL_GRAPH_001`
- Nodes: 1069
- Initializers: 1024, byte-identical to the source model
- Operators/boundaries: 215/215
- Output: `1x300x6`
- Feature lattice: 768, 384, 192, 96, 48, 24
- Deepest attention tokens: 576 (`N % 16 == 0`)
- MACs: 3,984,749,568
- FLOPs under the two-operations-per-MAC convention: 7,969,499,136
- Arena bytes: 11,796,480
- Peak live bytes: 7,077,888
- Package manifest SHA-256: `3fd4d004e92c4238c69c2d07bc1eedcee92c1968984f17d2b796b7bf01b4e0be`

ONNX checker and shape inference passed. Two independent graph generations and
two package generations produced byte-identical trees. All 215 qspec rows
common with R640 are unchanged; this is Q0, not recalibration.
