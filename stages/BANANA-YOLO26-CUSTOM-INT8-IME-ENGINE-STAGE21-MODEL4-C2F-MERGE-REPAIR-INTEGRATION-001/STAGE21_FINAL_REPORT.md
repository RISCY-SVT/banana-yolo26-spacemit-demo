# Stage21 Final Report

classification: `stage21-model4-c2f-c2-integrated-ready-for-next-repair-decision`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6ea3f0737c2063de94a7b4beac976180c4375872`
end_head: `d8025985bff6373aaf7082a47ad532a18bd64134`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`
selected_repair_from_stage20: `C2_split0_concat_lut_4t`
selected_mode: `Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT`

## Summary

Stage21 integrated the Stage20 C2 merge repair into the real `/model.4` C2f runner path as an explicit merge mode. The integration is local to `model4_c2f_runner` and does not change global backend dispatch.

The new real-runner mode is bit-exact against the in-process scalar/reference path:

```text
host CTest: 36/36 pass
board test_stage21_c2f_merge_repair: pass
concat_mismatches: 0
model4_cv2_mismatches: 0
```

The Stage20-compatible representative/full-shape transfer timing was replayed after integration:

```text
candidate: C2_split0_concat_lut_4t
mean_total_us: 116631
stddev_total_us: 364.855
mean_merge_us: 29767
mismatches: 0
threshold_3pct_us: 119828
transfer_gate: pass
```

## ONNX Oracle Caveat

Direct same-input integrated-runner vs full-model ONNX boundary comparison remains `partial`. Stage20's representative full-shape timing fixture repeats compact internal model4 tensors, while the Stage20 ONNX Runtime full-shape dump is generated from a full-model synthetic input. A direct comparison between those two output tensors would be invalid.

Stage21 records this explicitly in `engine_vs_onnx_fullshape_oracle_report.md` and recommends a Stage22 ONNX cut/subgraph oracle to close the same-input full-shape proof.

## Validation

```text
host_build: pass
host_ctest: pass
riscv_cross_build: pass
board_correctness: pass
board_stable_benchmark: pass
git_diff_check: pass
symlink_scan: pass
changed_only_secret_scan: pass
result_packet_export: pass
```

## Broken

```text
direct same-input full-shape ONNX cut for integrated runner: not yet closed
```

## Proven

```text
Stage20 C2 repair integrated into real model4 C2f runner as explicit mode.
Host compact C2 exactness passes.
Board compact integrated C2 exactness passes.
Stage20-compatible full-shape C2 timing remains within +3% gate.
No full engine, no graph scheduler, no ncnn mutation, no production claim.
```

## Unknown

```text
Whether a same-input ONNX cut will expose any full-shape boundary mismatch.
Whether /model.4/cv2 Conv should be tuned before graph expansion.
```

## Next Recommended Step

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001
```
