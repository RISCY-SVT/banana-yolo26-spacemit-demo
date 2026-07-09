# STAGE39 FINAL REPORT

classification: stage39-im2col-pack-partial-total-win-im2col-gate-missed
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c
end_head: pending-local-commit-see-final-response
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
selected_lane: branch3x3_im2col_pack_dataflow
selected_candidate: A1_A2_fast_chunks
selected_mode: Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK + Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE

## Correctness

- onnx_cut_status: pass
- mismatches: 0
- max_abs_diff: 0
- output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
- frm_sweep_status: pass
- affinity_ok: 1
- CPU4-7 IME execution: none

## Performance

| metric | Stage38 replay | Stage39 fastpack | speedup |
|---|---:|---:|---:|
| mean_total_us | 30334.500 | 28073.000 | 1.080558x |
| stddev_total_us | 207.007 | 192.905 | |
| combined_branch3x3_im2col_pack_us | 5558.070 | 5355.820 | 1.037763x |
| combined_branch3x3_conv_us | 10291.260 | 8882.390 | 1.158614x |
| no_measure_total_us | 30253.700 | 27977.900 | 1.081343x |

The candidate passes correctness and improves selected-cut total and combined branch 3x3 conv total, but misses the explicit `combined_branch3x3_im2col_pack_us >= 1.30x` gate. Therefore this is recorded as a partial local sidecar win, not a full im2col/pack repair closure.

## Post-Stage39 Buckets

- conv_share_pct: 56.674800
- activation_share_pct: 10.703800
- merge_share_pct: 7.451210
- output_quantize_share_pct: 16.202400
- attribution_pct: 99.894100

## Validation

- host build + CTest: pass, 42/42
- RISC-V cross build with Y26_K1X_ENABLE_IME=ON: pass
- board CPU0-3 correctness/benchmark: pass
- hygiene/export: pending at report generation time

## Non-Claims

This is not full YOLO26 inference. This is not model FPS. This is not full-image/camera performance. This is not COCO/mAP. This is not production/default-backend readiness.

## Next Recommended Step

BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001

Reason: selected-cut local micro-tuning is now delivering diminishing and instrumentation-sensitive gains. The next gate should prove a minimal full-model runner skeleton/dataflow plan before more selected-cut micro-optimizations.
