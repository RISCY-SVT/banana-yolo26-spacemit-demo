# STAGE33 FINAL REPORT

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `1304c765cbb97241a5ac5700bb91c0fce7d4c60b`
end_head: `final-head-copy-in-result-packet-after-local-commit`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Classification

`stage33-mixed-signedness-correct-but-regresses`

## Selected Candidate

candidate: `smt.vmadotus u8 x s8 fused-correction for /model.4/cv2/conv/Conv`
status: `correct-but-rejected-for-performance`

The candidate is available only as an explicit Stage33 local diagnostic mode:

```text
Y26_STAGE16_MERGE_MODE_STAGE33_MODEL4_CV2_MIXED_SIGNEDNESS
bench merge option: branch1_add_lut_mixed_cv2
```

It is not selected as the accepted runner mode and does not change any global/default backend behavior.

## Scope

Selected path only:

```text
/model.4 same-input ONNX-cut path
target node: /model.4/cv2/conv/Conv
shape: 80x80x96 -> 80x80x128
kernel: 1x1
```

This is not full YOLO26 inference, not model FPS, not full-image/camera performance, not COCO/mAP, and not production/default-backend readiness.

## Correctness

same_input_onnx_cut: `pass`
mismatches: `0`
max_abs_diff: `0`
output_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
frm_sweep: `pass`
ambient frm tested: `RNE RTZ RDN RUP RMM`
post_call_frm_restored: `yes`
board_affinity: `CPU0-3`
CPU4_7_IME_execution: `none`

## Performance

Protocol:

```text
board: Banana-Pi BPI-F3 / SpacemiT K1X
pinning: taskset -c 0-3
warmup: 10
runs: 100
repeats: 5
```

| mode | total_us | stddev_us | model4_cv2_conv_us | model4_cv2_correction_us | model4_cv2_compute_us | model4_cv2_copy_us | status |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline `branch1_add_lut` | 40380.4 | 592.759 | 11852.7 | 1742.83 | 8129.4 | 0 | pass |
| candidate `branch1_add_lut_mixed_cv2` | 40934.1 | 446.612 | 12862.2 | 0 | 9699.05 | 1127.18 | correct-regressed |

Derived:

```text
model4_cv2_correction_us: 1742.83 -> 0
model4_cv2_conv_speedup: 0.9215x
selected_cut_total_speedup: 0.9865x
selected_cut_total_regression: 1.37%
```

Acceptance gates:

```text
A. correction_us drops by >=50% with no total regression >1%: fail
B. /model.4/cv2 total_conv_us improves by >=1.05x: fail
C. selected-cut total_us improves by >=1.02x: fail
```

Decision: do not select/promote the mixed-signedness candidate for this node.

## What Changed

- Added named-asm `smt.vmadotus` proof wrapper for `u8 x s8 -> s32`.
- Added a local MMT4D fused-correction `u8s8` path for `/model.4/cv2/conv/Conv`.
- Added explicit Stage33 runner mode and bench option for diagnostic A/B testing.
- Added host oracle test for the signed-storage baseline vs mixed-signedness algebra.
- Patched Stage32 traceability placeholders to the actual Stage32 head.

## Validation

```text
host_native_build: pass
host_ctest: pass (42/42)
riscv_cross_build_Y26_K1X_ENABLE_IME_ON: pass
board_CPU0_3_correctness: pass
board_stable_benchmark: pass
frm_rounding_regression: pass
same_input_onnx_cut: pass
```

## Evidence

```text
log_dir: /data/ncnn-logs/ai-team/2026-07-08_14-41-34/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001
baseline_log: run_logs/stage32_baseline_replay_board.log
candidate_log: run_logs/stage33_candidate_board.log
cpu0_3_oracle_log: run_logs/stage33_cpu0_3_oracle_smoke.log
disassembly_log: run_logs/asm_disassembly_vmadotus.log
```

## Next Recommended Step

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-THREAD-COPY-OR-OUTPUT-QUANTIZE-LOCAL-REPAIR-001`

Recommended focus:

```text
1. Reduce threaded Conv copy/thread overhead around /model.4/cv2/conv/Conv.
2. Recheck output QuantizeLinear only if same-session buckets show it remains material.
3. Avoid selecting mixed signedness for /model.4/cv2 unless future work removes the added compute/copy cost.
```
