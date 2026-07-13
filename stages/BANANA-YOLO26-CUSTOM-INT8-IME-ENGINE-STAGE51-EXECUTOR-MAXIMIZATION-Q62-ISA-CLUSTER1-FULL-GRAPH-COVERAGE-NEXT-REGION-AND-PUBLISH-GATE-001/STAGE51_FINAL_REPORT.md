# Stage 51 final report

classification: stage51-executor-maximized-next-region-strong-positive
publication_classification: post-commit parity recorded in result packet
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE51-EXECUTOR-MAXIMIZATION-Q62-ISA-CLUSTER1-FULL-GRAPH-COVERAGE-NEXT-REGION-AND-PUBLISH-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: ea993fb4255f12592380b975bd3cc6dbc73bea57
end_head: self-identity recorded in result packet by design
commit_created: true after this tree is sealed
pushed: final fast-forward verification recorded outside this tracked tree

## Proven

- The exact Stage50 commit regressed cleanly: model5 4826.757964 us mean / 4918.032450 us p95; model4-final to model8 27261.068684 us / 27396.073300 us.
- A 10,000-run soak measured p99 30232.527000 us and p99.9 33049.680200 us.
- Exact Q62 E2c uses explicit `vsmul.vv` e64, preserves K1X_INT8_V1 and vector CSR state, and reduces model5 to 3658.164866 us and the model4-final to model8 slice to 17828.345456 us.
- U0 remains the evidence-backed ISA/compiler contract; broader common extensions and explicit `-mcpu` were exact but slower.
- Cluster1 non-IME execution is exact with zero CPU4-7 IME, but is not selected because it regressed the slice.
- Full-graph stable full-shape MAC coverage is 96.172127%; all material non-MAC classes are measured or conservatively mapped.
- Exactly one region was implemented: model8 output through complete model9 SPPF/residual. It is exact and strong-positive at 3455.120892 us mean / 3514.509150 us p95 versus ORT 19115.925104 / 19123.943940 us.
- The region and combined persistent path use zero internal conversions and zero float materializations.

## Broken or rejected

- U1/U2 common-extension builds, cluster1 offload, hugepage-only memory policy, and IRQ retargeting did not beat the selected route.
- Named `_xsmtvdot` parser spelling is unavailable; the proven explicit IME assembly remains selected without a raw-opcode lane.
- X60 named cache/stall PMU events remain unavailable; unsupported counters are not reported as zero.

## Unknown

- Full-model custom latency and full-model K1X_INT8_V1 COCO accuracy are not measured.
- Small-N head, grouped/depthwise Conv, attention MatMul/Softmax, Resize, and final selection still use conservative estimator rows rather than production custom implementations.
- Production readiness, achieved 20 FPS, camera performance, and trained-student accuracy remain unproven.

## Decision

The current graph is not target-credible on the maximized measured substrate: analytical envelopes
are 158.973694 / 204.380817 / 269.869364 ms, all
well above the 45 ms pure-model target. Preserve the exact executor evidence and route the next
separately authorized stage to model-executor co-design preparation with both 416 and 512 held.
