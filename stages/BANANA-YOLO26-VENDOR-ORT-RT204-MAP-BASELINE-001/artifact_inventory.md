# artifact_inventory

stage_id: BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: b54c8767e691dc57cbd035a13d2d2d348d2f5366
current_head_at_report_write: b54c8767e691dc57cbd035a13d2d2d348d2f5366
log_dir: /data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
shared_log_dir: /data/ncnn-logs/ai-team/2026-07-08/2026-07-08_04-06-58__contcodex__BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001__trackb-yolo26-ort-map

## Key generated artifacts

| artifact | path | sha256 |
|---|---|---|
| FP32 COCO predictions | /data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/artifacts/coco/predictions_fp32_e2e_rt204.json | bb815dc211cb763043a8b148be08144bdaf651a2edf5e097ab351e8915599f31 |
| FP16 keep-I/O COCO predictions | /data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/artifacts/coco/predictions_fp16_keepio_rt204.json | ca588202512394856f5bd2a55b8363fcba93f0621c3a9a7e9dace59da7341c63 |
| COCO mAP matrix TSV | /data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/artifacts/tables/coco_map_matrix.tsv | local artifact |
| FPS matrix TSV | /data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/artifacts/tables/fps_matrix.tsv | local artifact |
| Output sanity table TSV | /data/ncnn-logs/ai-team/2026-07-08_06-06-54/BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001/artifacts/tables/output_contract_sanity.tsv | local artifact |

Large prediction JSON files are intentionally stored in the run/result artifacts, not tracked in git.
