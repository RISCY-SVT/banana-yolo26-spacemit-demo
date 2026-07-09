# Stage36 Final Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE36-CV2-PIPELINED-VMADOT-CANDIDATE-001
classification: stage36-cv2-pipelined-vmadot-selected
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: a8b76072f19ff792bc5afc33ab93a022f2c26eb6
end_head: pending-local-commit-see-final-response
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Selected Candidate

selected_candidate: A1_branch1_add_lut_cv2_pipelined4

Stage36 added an explicit local model4 cut mode:

- `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4`
- `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED6`

The accepted candidate is the 4-accumulator signed-storage `s8 x s8 -> s32` `smt.vmadot` path for `/model.4/cv2/conv/Conv`. It preserves the Stage26 branch1/add LUT path, existing explicit correction semantics, and the same ONNX-cut output boundary.

The 6-accumulator candidate was byte-correct but slightly slower than the 4-accumulator candidate in the same session, so it remains diagnostic only.

## Correctness

same_input_onnx_cut_status: pass
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
board_cpu0_single_thread_status: pass
board_cpu0_3_threaded_status: pass
frm_sweep: pass for RNE, RTZ, RDN, RUP, RMM
affinity_ok: 1
cpu4_7_ime_execution: none

## Performance

Stable protocol:

- board: Banana-Pi BPI-F3 / SpacemiT K1X / X60
- affinity: `taskset -c 0-3`
- warmup: 10
- runs: 100
- repeats: 5
- timing source: accepted steady-clock runner timing; no `rdcycle`

| mode | mean_total_us | stddev_total_us | cv_pct | model4_cv2_compute_us | model4_cv2_conv_us | mismatches | sha_status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0_branch1_add_lut | 37341.1 | 405.161 | 1.08503 | 7541.75 | 10420.4 | 0 | match |
| A1_cv2_pipelined4 | 33192.7 | 364.104 | 1.09694 | 3616.14 | 6307.54 | 0 | match |
| A2_cv2_pipelined6 | 33217.5 | 418.787 | 1.26074 | 3822.91 | 6521.77 | 0 | match |

Performance gates:

- A1 `model4_cv2_compute_us` speedup: 2.085580x
- A1 selected-cut total speedup: 1.124979x
- A1 `model4_cv2_conv_us` speedup: 1.652055x

Stage36 therefore passes both the minimum and good total gates, and reaches the excellent `model4_cv2_compute_us` speedup gate.

## Bucket State After Selection

For A1:

- conv_share_pct: 56.9509
- activation_share_pct: 8.79814
- merge_share_pct: 6.48432
- output_quantize_share_pct: 19.8476
- mean_thread_overhead_us: 4362.11
- mean_attribution_pct: 99.9195

The selected-cut path remains Conv-dominant overall, but `/model.4/cv2/conv/Conv` raw compute is no longer the same bottleneck it was before Stage36. The remaining Conv work is mostly the branch 3x3 paths and thread/correction overhead. Output quantize is also material.

## Validation

host_build: pass
host_ctest: pass, 42/42 tests passed
riscv_cross_build: pass with `Y26_K1X_ENABLE_IME=ON`
board_correctness: pass
board_stable_benchmark: pass
git_diff_check: pass
symlink_scan: pass, no symlinks under `custom_int8_engine` or `stages`
secret_path_scan: pass
result_packet: pending-export

## Non-claims

This is not full YOLO26 inference.
This is not model FPS.
This is not full-image or camera performance.
This is not COCO/mAP.
This is not production or default-backend readiness.

## Next Recommended Step

Open Stage37 as a selected-cut next-bottleneck gate:

BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

Stage37 should keep the Stage36 A1 mode as baseline, rebuild per-bucket attribution, and choose exactly one local lane: branch 3x3 Conv/thread-overhead repair, output QuantizeLinear repair, or stop if no local selected-cut lane has a credible gain.
