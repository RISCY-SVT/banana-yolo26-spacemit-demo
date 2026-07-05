# Stage 14 Sidecar Candidate Review

Stage 14 reviewed Stage 13 merge/dataflow sidecars before expanding to `/model.3` and `/model.4/cv1`.

## Candidates

| candidate | status | evidence |
|---|---|---|
| `A0_stage13_replay` | baseline | materialized Stage 13 path |
| `A1_split0-cache + fused add/concat` | correctness-equivalent option | one CPU0 replay measured lower total than A2 (`433682 us` vs `436117 us`) |
| `A2_post_concat_qdq_rvv` | selected Stage 14 sidecar default | explicit RNE RVV post-Concat QDQ; preserves accepted `A2_fused_qdq_nhwc` handoff |

## Stage 15 Policy

Stage 15 must not blindly force Stage 14 `A2_post_concat_qdq_rvv` for every future boundary.

Per-boundary selection order:

1. correctness and ONNX oracle equivalence;
2. total time and component time;
3. maintainability.

For the Stage 15 primary target, no new `/model.4` Add/Concat merge is included. The selected boundary reaches only:

`/model.4/cv1/act` -> `/model.4/Split` -> `/model.4/Split_output_1` Q/DQ -> `/model.4/m.0/cv1/conv/Conv` -> `/model.4/m.0/cv1/act` Q/DQ.

Therefore the Stage 13 A1/A2 merge sidecar is retained as background evidence, not applied to a new merge in Stage 15.
