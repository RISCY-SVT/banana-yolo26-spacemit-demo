# Cross-runtime output0 diagnostic

This compares one deterministic input across different ORT runtimes. It is not COCO/mAP or model accuracy.

- host_raw: `.deps/custom_int8_engine/stage42_contract_repair/host_disable/tensors/10_output0.bin`
- host_sha256: `45856c19c94d285cc3e847b801c30eff5107cbd2e9bced60ac8578707fd752a4`
- board_raw: `/data/ncnn-logs/ai-team/2026-07-10/2026-07-10_06-32-46__codex__BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE42-INPROCESS-ORT-CONTRACT-REPAIR-AND-MODEL16-ORACLE-GATE-001__stage42-technical-rerun/artifacts/board/disable_boundaries/board_10.bin`
- board_sha256: `f41d00ca940d92cb0a8b67631f0e3d506408ffeb95e3be02bb60028f65a8b75a`
- exact_rows_same_index: 0/300
- same_class_same_index: 16/300
- class_count_l1_distance: 80

## Score distribution

| runtime | min | p50 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| host | 0.000037432 | 0.000062555 | 0.000395036 | 0.000825009 | 0.010168414 | 0.027492136 | 0.000486562 |
| board | 0.000026673 | 0.000052899 | 0.000479937 | 0.002208769 | 0.016809573 | 0.044935435 | 0.000725662 |

## Top-k class multiset overlap

| k | overlap | ratio |
|---:|---:|---:|
| 10 | 9 | 0.900000 |
| 50 | 39 | 0.780000 |
| 100 | 85 | 0.850000 |
| 300 | 260 | 0.866667 |

## Greedy class+IoU matching (IoU >= 0.50)

- matched_rows: 222/300
- mean_iou: 0.902614167
- p50_iou: 0.936675106
- mean_abs_coordinate_diff: 6.188908100
- max_abs_coordinate_diff: 231.901763916
- mean_abs_score_diff: 0.000396177
- max_abs_score_diff: 0.017443299
