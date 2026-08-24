# Provider subgraphs

B2: one fused subgraph, 925 nodes, SHA-256 `16811f81e7212d80bf7bf580dbe2f15b07c6f445212e211d4a511881b86c8edb`.

C2: one fused subgraph, 925 nodes, SHA-256 `d57aea4d58ce9725edfac537515536f76a77f3863de4786e14342daa6e642997`.

Partition topology equality (node count, graph I/O and op-type census): `true`. Different graph bytes are expected because C2 changes frozen qparam initializers. The profiler exposes one SpaceMIT fused event and zero CPU inference events for each EP session; the separate float tail is intentionally CPU.
