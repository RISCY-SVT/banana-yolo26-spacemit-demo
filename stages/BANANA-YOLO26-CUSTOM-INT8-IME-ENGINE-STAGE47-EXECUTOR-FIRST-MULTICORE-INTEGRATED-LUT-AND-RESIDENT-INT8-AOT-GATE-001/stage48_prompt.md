
# Stage48: INT8 Semantic Contract and Student Architecture Preparation Gate

Resolve exactly one decision lane: define a graph/export contract whose Conv and
requant semantics are exact integer operations on host and K1X, then use the
Stage47 measured LUT to specify both 416 latency-primary and 512 accuracy-oriented
student candidates. Do not train, select a final resolution, reopen RT205, add an
ISA lane, or implement a production engine. Require independent integer oracles,
fixed-host replay, K1X scalar/IME parity, and an explicit operator/layout/quant
contract before any architecture-training authorization.
