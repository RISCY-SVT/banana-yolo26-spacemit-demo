# Conv Strategy Decision

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001

## Selected Decision

```text
DECISION_B_MMT4D_MAINLINE_SIGNEDNESS_OR_CORRECTION_NEXT
```

## Classification

```text
stage32-mixed-signedness-proof-ready-for-mmt4d-correction-stage
```

## Why Not Direct/Sliding

The Stage31 direct/sliding sidecar is correct but still slower:

```text
direct_sliding_vmadot123_stage31_replay_us: 58831.5
mmt4d_1thread_stage31_replay_us: 22141.0
mmt4d_4thread_stage31_replay_us: 5916.17
```

The required low-overhead layout gate did not pass:

```text
required_attachable_panel_build_us: <= 7800
best_attachable_stage32_layout_us: 18447.2
best_attachable_candidate: B3_interior_fast_path
```

Therefore no direct/sliding integration is justified.

## Why MMT4D Mainline Remains Selected

Current selected-cut replay remains byte-exact:

```text
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
attribution_pct: 99.9412
```

The Conv bucket is still dominant:

```text
conv_us: 25988.2
conv_share_pct: 63.9366
```

The current path already uses IME through plain `smt.vmadot`; it must not be described as non-IME.

## Why Signedness/Correction Is the Next Lane

Stage32 proved the integer dot signedness family by parser/disassembly/board/oracle:

```text
smt.vmadot:   s8xs8, mismatches=0
smt.vmadotu:  u8xu8, mismatches=0
smt.vmadotsu: s8xu8, mismatches=0
smt.vmadotus: u8xs8, mismatches=0
```

The correction bucket is bounded and measurable:

```text
aggregate_correction_us: 2497.31
model4_cv2_correction_us: 1758.66
```

This is not a guaranteed large win. It is the next safest proof because it can be evaluated locally against the already closed `/model.4` same-input ONNX cut without graph expansion.

## Rejected Decisions

```text
DECISION_A_LOW_OVERHEAD_SLIDING_NEXT: rejected, layout gate failed.
DECISION_C_MMT4D_MAINLINE_TILE_OR_THREAD_OVERHEAD_NEXT: deferred, raw compute dominates but no bounded tile candidate was proven in Stage32.
DECISION_D_STOP_CONV_MICROKERNEL_FOR_NOW: rejected, mixed signedness proof created one bounded next lane.
```
