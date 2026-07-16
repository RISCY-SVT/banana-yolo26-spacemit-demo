# Co-design Seed V4

This directory is an evidence bundle for a future, separately authorized
model-executor co-design project. Stage57 did not select a student, change the
model, train a model, or create a co-design branch.

## Frozen current-graph basis

- Integer contract: `K1X_INT8_V1`
- Graph profile: `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001`
- Model SHA-256: `30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c`
- Package-manifest SHA-256: `fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be`
- Accepted prediction SHA-256: `cda5c8c7a46d61d9c90f6292001eea190cb8f6617efe647a33dc6134dd57ccda`
- Layout: `NCHWc8_SPATIAL_INNER_V1`

The current-graph executor branch is frozen after Stage57. Reusing this seed
requires a separate branch/project and explicit authorization.

## Contents

- `contracts/`: complete graph, operation whitelist, layout, and quantization
  contracts for the frozen model.
- `cost-model/`: measured latency rows, V4 shape validation, current-graph cost
  model, and the Stage56 reserve ledger.
- `kernel-evidence/`: selected and rejected source-level kernel contracts and
  decisions. A rejected candidate is not a proof that every related schedule is
  slower.
- `hpm/`: measured HPM interpretation and its limitations.

## Required use policy

1. Preserve exact identities with every copied result.
2. Directly measure any novel shape marked `high_uncertainty=yes`.
3. Treat the V4 model as a mean-latency estimator, not a tail model.
4. Do not infer cache miss ratios when access counters were unavailable.
5. Do not interpret backend-stall events as proof of one dependency chain.
6. Revalidate every candidate composition on the physical K1X board.

The held-out V4 validation has a 5.660626% median absolute percentage error,
20.083997% p90, and 95.547103% worst case. The worst case is the 512-resolution
`dense_1x1_k32` class and is explicitly direct-measure-required.

No training or student-selection result is included because neither operation
was authorized or performed.
