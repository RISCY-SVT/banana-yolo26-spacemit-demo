# STAGE38 FINAL REPORT

classification: stage38-output-quantize-rvv-direct-selected-ready-for-branch3x3-im2col-pack-repair
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 97d9e8ce52c584a5ecc3d3aa44dc6e18e4e9e8a8
end_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
drive_verify_status: not-run-by-codex

## Selected Lane

- selected_lane: `A`
- selected_candidate: `A2_fuse_clamp_store_remove_intermediate_buffer`
- selected_mode: `Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4 + Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE`

## Correctness

- onnx_cut_status: `pass`
- mismatches: `0`
- max_abs_diff: `0`
- output_sha256: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`
- frm_sweep_status: `pass`
- affinity_ok: `1`
- CPU4-7 IME execution: `none`

## Performance

| metric | Stage37 replay | Stage38 selected | speedup |
|---|---:|---:|---:|
| mean_total_us | 32890.5 | 30341.5 | 1.08401x |
| stddev_total_us | 269.396 | 281.576 | |
| output_quantize_us | 7055.2 | 4551.97 | 1.54994x |
| conv_us | 18051.0 | 18048.1 | 1.00016x |
| activation_requant_us | 3004.23 | 3036.11 | 0.98950x |
| merge_us | 2112.38 | 2075.46 | 1.01779x |

## Bucket Attribution

- bucket_attribution_status: `pass`
- Stage37 replay attribution: `99.9173%`
- Stage38 candidate attribution: `99.9083%`
- im2col_split_status: `pass`
- thread_overhead_us is diagnostic and included inside `conv_us`; it was not double counted.

## Post-Stage38 Bottleneck

After Lane A:

- conv_share: `59.4832%`
- output_quantize_share: `15.0024%`
- activation_requant_share: `10.0064%`
- merge_share: `6.84031%`
- combined branch 3x3 im2col_pack_us: `5601.91`
- branch 3x3 im2col share of combined branch conv: `54.67%`

## Validation

- host build: `pass`
- host CTest: `pass`, `42/42`
- RISC-V cross build with `Y26_K1X_ENABLE_IME=ON`: `pass`
- board CPU0-3 correctness: `pass`
- board stable benchmark: `pass`
- result packet: pending export

## Non-Claims

This is not full YOLO26 inference. This is not model FPS. This is not full-image/camera performance. This is not COCO/mAP. This is not production/default-backend readiness.

## Next Recommended Step

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001`
