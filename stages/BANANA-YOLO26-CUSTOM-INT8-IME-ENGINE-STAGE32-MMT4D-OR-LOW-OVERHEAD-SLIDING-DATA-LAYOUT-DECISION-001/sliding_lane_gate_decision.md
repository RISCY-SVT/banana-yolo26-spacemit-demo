# Sliding Lane Gate Decision

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001

## Decision

```text
direct_sliding_lane_rejected_panel_build_too_expensive
```

## Evidence

Stage31 replay:

```text
direct_total_us: 58831.5
panel_build_us: 40512.2
kernel_compute_us: 15872.5
mmt4d_1thread_us: 22141.0
mmt4d_4thread_us: 5916.17
```

Stage32 layout-only gate:

```text
required_attachable_panel_build_us: <= 7800
best_attachable_candidate: B3_interior_fast_path
best_attachable_panel_build_us: 18447.2
gate: fail
```

## Reason

The Stage31 direct/sliding failure is still dominated by data layout. Stage32 did not find an attachable low-copy A-window layout that makes the direct `smt.vmadot1/2/3` sidecar viable. The descriptor-only prototype is too cheap to ignore, but it is not currently attachable to the direct kernel without reintroducing gather or panel materialization work.

## Policy

Do not integrate the Stage31 direct/sliding kernel into the selected `/model.4` runner path.

Do not run a broad direct-conv rewrite in the next stage. The selected mainline remains threaded MMT4D with plain `smt.vmadot`.
