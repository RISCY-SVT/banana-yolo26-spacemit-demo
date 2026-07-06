# STAGE24_FINAL_REPORT

classification: `stage24-merge-dataflow-repaired-ready-for-conv-thread-tile-decision`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE24-BRANCH1-ACTIVATION-LUT-OR-MERGE-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `fce411e20eb649e7f7f0cfe65573848c0e8a1fd4`
end_head: `pending-local-commit-see-final-response`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Summary

Stage24 replayed the Stage23 same-input `/model.4` ONNX-cut runner API path, rebuilt non-overlapping bucket attribution, selected Lane B, and integrated one local merge repair:

```text
selected_lane: B
selected_candidate: B3_split1_concat_lut_scalar_add
onnx_cut_status: pass
runner_api_status: pass
rounding_regression_status: pass
bucket_attribution_status: pass
host_tests: pass
board_tests: pass
mismatches: 0
max_abs_diff: 0
```

## Timing

| path | total_us | stddev_us | merge_us | conv_share_pct | activation_share_pct | merge_share_pct | output_quantize_share_pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage23 replay baseline | 147624 | 113.47 | 43334.4 | 42.0763 | 22.0935 | 29.3546 | 4.69731 |
| Stage24 B3 | 125229 | 147.027 | 20953.9 | 49.5835 | 26.0505 | 16.7325 | 5.54777 |

```text
merge_speedup: 2.06808x
total_speedup: 1.17883x
mean_attribution_pct: 99.9874
actual_output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
```

## Non-Claims

This stage is not full YOLO26 inference, model FPS, full-image/camera performance, COCO/mAP, production readiness, or default backend readiness.

## Next

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001`
