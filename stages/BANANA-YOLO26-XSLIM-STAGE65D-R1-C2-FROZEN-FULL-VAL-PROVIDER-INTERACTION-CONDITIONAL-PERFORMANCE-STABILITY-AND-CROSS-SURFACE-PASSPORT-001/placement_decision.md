# Stage65D-R1 placement decision

Decision: `pass`. B2 and C2 each expose one SpaceMIT fused inference subgraph with `925` source nodes, equal graph I/O and op census, and zero unexpected CPU inference events. The separate common float tail remains intentional CPU work.
