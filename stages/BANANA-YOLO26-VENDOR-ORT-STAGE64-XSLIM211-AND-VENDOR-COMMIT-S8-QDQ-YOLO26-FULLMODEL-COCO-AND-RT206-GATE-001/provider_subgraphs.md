# Provider subgraphs

## Selected official route

The selected `R211_PROJECT_EXACT_SPLIT` model is executed as two explicit
sessions:

| Component | Nodes | Provider contract |
|---|---:|---|
| S8-QDQ inference graph | 1,161 | SpacemiT EP decision surface |
| floating-point post-processing tail | 34 | CPU EP by design |

The inference graph contains 102 Conv, four MatMul, 354 QuantizeLinear, and 458
DequantizeLinear nodes. ORT profiling records the complete measured inference
work as one `SpaceMITExecutionProvider` Spine subgraph. No CPU provider event
was observed in the inference-session profile.

The transformed profile exposes a fused subgraph event, not an auditable
source-node-to-provider map. Consequently:

- meaningful Conv/MatMul placement is established at the complete inference
  subgraph boundary;
- the report does not claim individual source-node durations;
- the separate CPU tail is intentional and excluded from unexpected fallback;
- unexpected inference fallback is reported as zero observed profile events,
  not as proof about internal closed-provider implementation details.

The literal-preprocessing candidate shows the same one-subgraph placement
shape. The vendor-reference split is reused only when its generated model and
split artifacts are byte-identical to an already measured official candidate.
