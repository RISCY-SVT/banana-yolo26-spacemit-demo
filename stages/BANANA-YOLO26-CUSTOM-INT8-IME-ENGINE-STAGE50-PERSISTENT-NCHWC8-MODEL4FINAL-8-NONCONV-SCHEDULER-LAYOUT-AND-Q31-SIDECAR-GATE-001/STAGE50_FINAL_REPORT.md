# Stage 50 final report

classification: stage50-model4final8-strong-net-positive
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE50-PERSISTENT-NCHWC8-MODEL4FINAL-8-NONCONV-SCHEDULER-LAYOUT-AND-Q31-SIDECAR-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: eb8e28194853d8f70c8a8a8d008253396327aac1
end_head: recorded-post-commit-in-result-packet
commit_created: true
commit_hash: recorded-post-commit-in-result-packet
pushed: false

## Proven

- Stage 49 regressions passed before edits: model5 4916.347994 us mean / 5010.036300 us p95; model4-final to model6 26633.364540 us / 26730.553250 us.
- The deterministic K1X_INT8_V1 model4-final to model8 package has manifest SHA-256 `2dbdbd18abe1ba126f12246b82c25821b9f74eb0ee9c324cb30aaaa062f64527`, 32 tensor boundaries, and 29 operations.
- Python arbitrary precision, portable C++ scalar, board scalar, and board IME are exact at all 32 boundaries for F0-F7.
- FRM sweep passes and restores ambient mode; IME is CPU0-3 only; CPU4 is controller-only; no SIGILL occurred.
- Persistent custom internal mean is 27174.621676 us and p95 is 27269.385650 us.
- Equivalent B120 ORT diagnostic mean is 57909.967262 us and p95 is 58031.158848 us.
- Custom deltas are -53.074362% mean and -53.009062% p95, with zero internal layout conversions and zero float materializations.
- Stable headline timing binary SHA-256: `f404c962ec3ee1e4613d2f65d05574cb811da82345220d344e0a4f91deb96f88`. Final validation/PMU-accounting binary SHA-256: `cf46e61f678390ba31b917cdaa3fa3b8db42da859b0584357eb09863e24edd8f`; the intervening source change only accumulates diagnostic counter enabled/running time and is inactive when no counter is requested.

## Broken or rejected

- Adapter-inclusive diagnostic is 62920.626352 us and remains slower than ORT; per-island adapters are rejected as the architecture.
- Complete exact explicit-vector Q62 E2b was not achieved; E1 scalar Q62 remains selected.
- Q31 is exact to its own contract but is a performance no-win and differs from V1 in 42,820 compared elements; it is not promoted.
- Alternative layout contracts round-trip exactly, but no alternative persistent kernel pair met the required performance proof.

## Unknown

- Full-graph latency and full-model K1X_INT8_V1 COCO accuracy are not measured.
- X60 named cache/stall events remain unavailable; generic cache events had `time_running=0`.
- A complete production executor, camera pipeline, and achieved 20 FPS remain unproven.

## Decision

The main route remains K1X_INT8_V1 with NCHWc8, M12xN16 plus exact tail, P3 delivery, E1 exact Q62, four spatially partitioned CPU0-3 workers, explicit RVV LUT, and active-worker completion. Stage 51 should close full-graph LUT coverage and choose one next resident region; it must not claim production readiness or automatically start student training.

The exact commit identity is recorded in the exported result packet after commit creation. Embedding a commit's own hash in its tracked tree would change that hash.
