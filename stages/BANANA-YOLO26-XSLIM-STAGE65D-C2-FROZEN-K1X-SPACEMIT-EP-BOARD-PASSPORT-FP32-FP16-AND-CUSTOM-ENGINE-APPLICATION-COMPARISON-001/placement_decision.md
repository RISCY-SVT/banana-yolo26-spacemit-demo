# Placement decision

Decision: `pass`.

B2 and C2 each compile into one SpaceMIT inference subgraph with `925` source nodes. The graph I/O and op-type census are equal, and the profiled EP sessions contain zero unexpected CPU inference events. The separate common float tail remains intentional CPU work.

Different provider-dump bytes reflect the frozen C2 qparam initializer changes; they are not treated as partition drift.
