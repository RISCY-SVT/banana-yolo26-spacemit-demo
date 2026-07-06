# STAGE20 FINAL REPORT

classification: `stage20-model4-fullshape-repaired-ready-for-next-step`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE20-ACTIVATION-REQUANT-AND-MODEL4-FULLSHAPE-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `53ac15ad253ac70e594cc7e1ac6c117e92da85ca`
end_head: `pending-local-commit-see-result-packet-final-head-copy`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Selected Scope

selected_subset: `candidate_K_model4_c2f_representative_fullshape_synthetic`
fullshape_oracle_status: `pass`
fullshape_timing_status: `pass`
selected_repair_lane: `C2`
selected_repair_status: `pass`

Stage20 created a reusable host-side full-shape ONNX Runtime extractor and used representative/full-shape model4 C2f timing to choose a single local repair lane. The measured bottleneck was merge/post-Concat-QDQ, so Stage20 selected C2 and tested `C2_split0_concat_lut_4t`.

## Key Results

```text
B1_threaded_branch0_4t:
  mean_total_us: 149539
  stddev_total_us: 76.3189
  cv_total_pct: 0.0510361
  mean_merge_us: 66564.3
  merge_share_pct: 44.513

C2_split0_concat_lut_4t:
  mean_total_us: 116338
  stddev_total_us: 121.933
  cv_total_pct: 0.104809
  mean_merge_us: 29791.6
  merge_share_pct: 25.6078
  mismatches: 0
```

The selected repair improved representative/full-shape selected-subset total time by about `1.2854x` versus the pre-repair 4-thread branch0 candidate.

## Validation

host_tests: `pass`
board_tests: `pass`
cross_build: `pass`
mismatches: `0`

Host CTest passed `35/35`. RISC-V cross build with `Y26_K1X_ENABLE_IME=ON` passed. Board Stage20 stable benchmark used `taskset -c 0-3`, `warmup=10`, `runs=100`, `repeats=5`; all candidates reported `mismatches=0` and `affinity_ok=1`.

## Non-Claims

This is not full YOLO26 inference, not full-image or camera performance, not COCO/mAP, not model FPS, and not production/default-backend readiness.

## Next Recommended Step

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`
