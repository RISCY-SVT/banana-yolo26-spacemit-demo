# Stage 54 Draft: Residual Dense-Conv Ceiling and Handoff-Default Gate

This is a draft next-stage recommendation. It does not authorize execution.
A separate direct user prompt must provide the final Stage53 HEAD and launch
authorization.

```yaml
task_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE54-RESIDUAL-DENSE-CONV-CEILING-STATIC-SCHEDULE-AND-HANDOFF-DEFAULT-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
expected_start_head: resolve-from-Stage53-final-report
target: Banana-Pi BPI-F3 / SpacemiT K1X
stage_kind: residual-executor-ceiling-and-handoff-policy-decision
direct_user_authorization: required
model_training_authorized: false
student_architecture_selection_authorized: false
model_executor_codesign_authorized: false
rt205_work_authorized: false
q31_main_contract_promotion_authorized: false
cpu4_7_ime_authorized: false
production_claim_authorized: false
```

## Accepted Stage53 Basis

Stage53 preserves exact `K1X_INT8_V1`, full COCO output, and the complete
`1x300x6` graph while materially reducing full-model latency. Its calibrated
profile accounts for more than 99% of outer wall time. The dominant residual
category is dense Conv outside model4-final through model9:

```text
dense Conv outside resident region:
  136.284 ms
  56.787% of profiled wall

grouped/depthwise:
  19.660 ms

resident model4-final -> model9:
  19.468 ms

attention MatMul + exact Softmax/transpose:
  21.999 ms
```

Stage53 also leaves the condition-variable route as the compatibility default
and selects epoch-spin only as an opt-in optimized-research route because it
uses materially more process CPU.

## Mission

1. Build a stable per-shape dense-Conv LUT from the complete selected graph,
   including cache residency, packed-weight traffic, borders, M tails, and
   exact E2c2 store work.
2. Compare bounded M/N blocking and weight-stationary routes for the actual
   high-wall shapes. Do not extrapolate model5 throughput to other shapes.
3. Revisit Conv-to-LUT fusion only with a lifetime-safe destination contract;
   retain the Stage53 rejection if complete-model wall does not improve.
4. Implement and measure one prepared schedule-level batching candidate for
   consecutive short operations. Keep arithmetic and dependency barriers exact.
5. Decide whether epoch-spin is suitable for a documented dedicated-board
   profile or remains benchmark-only. Condition-variable `SCHED_OTHER` stays
   the portable handoff default unless complete soak and CPU-occupancy evidence
   supports a versioned policy change.
6. Recalibrate the measured full-graph cost model and update the optimized
   research bundle only if complete-model correctness, COCO identity, tail
   latency, and API compatibility all pass.

## Hard Gates

```text
- full 215-boundary F0-F7, bus, and Zidane exactness
- FRM and vector CSR restoration
- CPU0-3-only IME; CPU4-7 IME count zero
- selected symbol disassembly
- full-model ABBA timing, 10/100/5
- 10,000-run SCHED_OTHER soak for any selected handoff policy
- complete COCO val2017 if final predictions differ by one byte
- measured cost-model error <=15%
- normal fast-forward publication only
```

## Decision

Use complete-model SCHED_OTHER wall time, not isolated kernel throughput:

```text
residual-dense-strong-positive:
  >=15% full-model mean gain with no p95/p99 regression

residual-dense-positive:
  >=8% full-model mean gain with bounded tails

residual-dense-ceiling-reached:
  <8% full-model gain after exact measured candidates
```

If the ceiling is reached, the next human decision is either release
maintenance or a separately authorized model-executor co-design preparation
stage using the measured Stage53/54 cost model. This draft does not authorize
co-design, student selection, QAT, or training. It makes no 20 FPS or
production-readiness claim.
