# Stage 53: Full-Executor Hotspot Optimization and Release Maintenance Gate

```yaml
task_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE53-FULL-EXECUTOR-HOTSPOT-OPTIMIZATION-AND-RELEASE-MAINTENANCE-GATE-001
project: Banana-Pi BPI-F3 CROSSBUILD
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
expected_start_head: use the exact Stage52 final HEAD from its result packet
target: Banana-Pi BPI-F3 / SpacemiT K1X
stage_kind: measured-full-executor-hotspot-optimization-and-release-maintenance
direct_user_authorization: false
model_training_authorized: false
student_architecture_selection_authorized: false
model_executor_codesign_authorized: false
rt205_work_authorized: false
q31_main_contract_promotion_authorized: false
cpu4_7_ime_authorized: false
production_authorized: false
```

## Decision context

Stage 52 delivered the complete `K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001`
executor, deterministic package, full COCO result, C/C++ API, CLI, and
reproducible handoff bundle. Use the exact Stage 52 report, package hash,
release identity, full-model timing, operation profile, and soak as the
accepted baseline. Do not reopen the vendor-runtime lane.

This draft recommends further executor optimization and release maintenance.
It does not authorize co-design or training. A later co-design preparation
stage requires a separate user prompt after the current full executor has a
measured, reviewed optimization ceiling and an explicit accuracy target.

## Mission

1. Reproduce the Stage 52 package, F0-F7/real-image hashes, COCO prediction
   hash, safe-scheduler timing, and 10,000-run soak before edits.
2. Attribute complete full-model wall time with instrumentation disabled for
   headline measurements and profiling enabled only in separate runs.
3. Optimize only measured hotspots, ordered by full-model impact rather than
   model number. Candidate work is the dedicated RGB/RGBX stem, true N4/N8/N16
   kernels, explicit RVV grouped/depthwise Conv, integer attention
   MatMul/Softmax, E2c2, and static-schedule/barrier reduction.
4. Preserve `K1X_INT8_V1`, `NCHWc8_SPATIAL_INNER_V1`, SCHED_OTHER as the safe
   default, CPU0-3-only IME, zero internal float Q/DQ, and zero per-run
   allocation or file I/O.
5. Update the release only for exact, stable wins. Retain a rejected-candidate
   matrix so no local microbenchmark is promoted without a full-model gain.

## Gates

- Every optimized operator must match the independent integer oracle, portable
  scalar implementation, board scalar route, and selected board route exactly.
- Any arithmetic or final-output change requires F0-F7, real-image boundary
  checks, state-restoration tests, and full COCO val2017 revalidation.
- Candidate selection requires an improvement in complete SCHED_OTHER
  full-model mean with no material p95/p99 regression and a passing long soak.
- ORT remains a diagnostic comparator, not the integer semantic authority.
- Keep the release bundle reproducible across independently named build roots.

## Non-goals

```text
no training or QAT
no student 416/512 selection
no model-executor co-design implementation
no RT205/plugin work
no Q31 promotion
no CPU4-7 IME
no raw-opcode lane
no default demo/backend change
no production, camera, or 20-FPS claim without matching evidence
```

## Final routing

Choose one measured outcome:

```text
stage53-release-maintenance-complete
stage53-full-executor-optimization-positive
stage53-full-executor-optimization-no-win
stage53-full-executor-accuracy-regression
stage53-blocked-technical
```

The final recommendation may be continued executor maintenance, one more
narrow hotspot stage, or separately authorized co-design preparation. It must
not start co-design automatically.
