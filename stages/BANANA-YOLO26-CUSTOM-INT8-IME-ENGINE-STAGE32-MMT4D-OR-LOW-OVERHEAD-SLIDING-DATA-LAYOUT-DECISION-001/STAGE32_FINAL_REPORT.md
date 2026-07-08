# STAGE32 Final Report

classification: stage32-mixed-signedness-proof-ready-for-mmt4d-correction-stage
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 00aa667b8770cd9e6c7a5cdd24ac2714bb1d52a9
end_head: pending-local-commit-see-final-response
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Summary

Stage32 replayed Stage31, tested low-overhead sliding layout candidates, replayed the current selected MMT4D/threaded model4 cut, and audited the integer dot signedness family.

The direct/sliding `smt.vmadot1/2/3` lane remains rejected for now: no attachable layout candidate reached the required `<=7800 us` panel-build gate. The current mainline remains threaded MMT4D over plain `smt.vmadot`.

Stage32 did prove `smt.vmadot`, `smt.vmadotu`, `smt.vmadotsu`, and `smt.vmadotus` by parser/disassembly/board/oracle evidence. The selected next lane is a bounded MMT4D mixed signedness/correction proof.

## Stage31 Replay

| candidate | mean_us | stddev_us | mismatches |
|---|---:|---:|---:|
| direct_sliding_vmadot123_stage31 | 58831.5 | 26.6264 | 0 |
| mmt4d_1thread_stage31_replay | 22141.0 | 105.014 | 0 |
| mmt4d_4thread_stage31_replay | 5916.17 | 56.8996 | 0 |

Direct sidecar decomposition:

| bucket | mean_us |
|---|---:|
| panel_build_us | 40512.2 |
| kernel_compute_us | 15872.5 |
| correction_us | 796.151 |
| writeback_us | 856.234 |

## Low-Overhead Sliding Layout Gate

| candidate | attachable | mean_us | gate |
|---|---:|---:|---|
| B0_stage31_full_panel | 1 | 44704.7 | fail |
| B1_row_cache_materialized | 1 | 47224.4 | fail |
| B2_descriptor_only | 0 | 166.498 | pass-not-attachable |
| B3_interior_fast_path | 1 | 18447.2 | fail |
| B4_row_cache_descriptor_model | 0 | 38781.2 | fail |

Gate:

```text
required_attachable_panel_build_us: <= 7800
result: fail
direct/sliding integration: not attempted
```

## Current MMT4D Selected Cut

Correctness:

```text
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
affinity_ok: 1
mean_attribution_pct: 99.9412
```

Buckets:

| bucket | mean_us | share_pct |
|---|---:|---:|
| total_us | 40646.8 | 100.0 |
| conv_us | 25988.2 | 63.9366 |
| activation_requant_us | 2988.19 | 7.3516 |
| merge_us | 2127.29 | 5.2336 |
| output_quantize_us | 7119.27 | 17.515 |
| input_adapter_us | 2399.96 | 5.904 |
| other_us | 23.9033 | 0.0588 |

Conv sub-buckets:

| node | total_conv_us | compute_us | correction_us |
|---|---:|---:|---:|
| /model.4/m.0/cv1/conv/Conv | 7489.04 | 5766.48 | 217.943 |
| /model.4/m.0/cv2/conv/Conv | 6430.69 | 4334.1 | 520.71 |
| /model.4/cv2/conv/Conv | 12068.4 | 8164.37 | 1758.66 |

## Integer Dot Signedness Family

| mnemonic | scalar hypothesis | parser/disassembly | board CPU0-3 | oracle |
|---|---|---|---|---|
| smt.vmadot | s8xs8 | pass | pass | pass |
| smt.vmadotu | u8xu8 | pass | pass | pass |
| smt.vmadotsu | s8xu8 | pass | pass | pass |
| smt.vmadotus | u8xs8 | pass | pass | pass |

`vmadotn` remains not authorized. Stage32 did not run raw opcode probing.

## Decision

```text
selected_next_lane: DECISION_B_MMT4D_MAINLINE_SIGNEDNESS_OR_CORRECTION_NEXT
next_recommended_step: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001
```

## Validation Status

```text
host_tests: pass, 41/41
riscv_cross_build: pass
board_tests: pass
git_diff_check: pass
git_diff_cached_check: pass
symlink_scan: pass
secret_scan: pass
path_control_scan: pass
```

Detailed validation results are recorded in `source_hygiene_report.md` and the result packet.

## Non-Claims

This is not full YOLO26 inference.
This is not model FPS.
This is not full-image/camera performance.
This is not COCO/mAP.
This is not production/default-backend readiness.
