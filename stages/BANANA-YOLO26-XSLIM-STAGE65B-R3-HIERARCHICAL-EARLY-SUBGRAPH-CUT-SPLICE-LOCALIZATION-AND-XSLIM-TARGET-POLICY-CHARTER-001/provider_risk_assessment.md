# Provider risk assessment

Policy A keeps the vendor-declared signed S8-QDQ representation and therefore has the lower structural risk, but changed qparams may alter SpacemiT partition/fusion behavior. Policy B introduces mixed-precision boundaries in model.23 and may split a fused subgraph, add conversions, or cause CPU fallback. Host accuracy does not prove K1X provider placement, latency, or stability; each policy needs later signed-QDQ conformance, provider profiling, fixed-fixture, COCO, performance, and soak gates.
