# 2.0.6 provider subgraphs

With `ORT_ENABLE_ALL`, FP32 and FP16 each produce one large executable
SpacemiT EP subgraph.

| Surface | Source nodes | Dumped EP graph nodes | Conv in EP graph | MatMul in EP graph | EP profile-time share |
|---|---:|---:|---:|---:|---:|
| FP32 | 453 | 380 | 102/102 | 4/4 | 99.3353853% |
| FP16 | 535 | 428 | 102/102 | 4/4 | 99.1816852% |

The remaining CPU events are primarily final shape, gather, TopK, and output
housekeeping. Profile event counts are runtime events, not a one-to-one source
node assignment.

The INT8 provider emits a transformed 1,076-node diagnostic subgraph containing
98 Conv, four `ConvWithBinary`, four MatMul, 248 QuantizeLinear, and 452
DequantizeLinear nodes. It then aborts during provider compilation at the first
quantized Conv. That dump proves an attempted partition, not executable
placement; executed INT8 EP node count is zero.

`ORT_DISABLE_ALL` also runs for FP32 and, unlike 2.0.5, for FP16. It produces
additional/smaller FP16 EP subgraphs and more CPU housekeeping. These are
diagnostic optimization surfaces, not the selected timing surface.
