# Provider subgraphs

B2: one fused subgraph, 925 nodes, SHA-256 `1dab0f96311cf963259e4941f0761971b818a37df3a19348f314a37600f87e6d`.

A1: one fused subgraph, 925 nodes, SHA-256 `37e392768dc73fac5023924048d85adac121b86ce2058a8bf1fba83ed3f0dca9`.

Partition topology equality (node count, graph I/O and op-type census): `true`. Different graph bytes are expected because A1 changes frozen qparam initializers. The profiler exposes one SpaceMIT fused event and zero CPU inference events for each EP session; the separate float tail is intentionally CPU.
