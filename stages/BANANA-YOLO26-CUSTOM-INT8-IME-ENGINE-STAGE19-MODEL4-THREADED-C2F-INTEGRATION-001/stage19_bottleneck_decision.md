# Stage19 Bottleneck Decision

Decision inputs:

```text
Stage18 representative A4 activation_share_pct: 44.971379
Stage18 representative A4 conv_share_pct: 53.284180
Stage19 compact A4 total_speedup_vs_A0: 0.656284x
Stage19 compact A5 total_speedup_vs_A0: 0.402947x
Stage19 compact A5 activation_share_pct: 46.935998
```

Decision:

```text
next_recommended_step: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001
```

Rationale:

```text
The representative branch-entry Conv threading result remains strong.
The compact C2f integration is correct, but thread overhead dominates compact execution.
Activation/requant is already the largest non-Conv bucket after representative Conv threading.
The naive row-parallel activation sidecar is exact but not useful on compact tensors.
Stage20 should first establish representative/full-shape model4 C2f timing, then repair activation/requant fusion or memory/dataflow using that evidence.
```

Rejected next steps:

```text
next graph expansion: premature without representative/full-shape model4 C2f timing
vmadot1/2/3 implementation: not authorized here and not justified by Stage19 compact overhead
full engine/default backend: out of scope
COCO/mAP/camera: out of scope
```
