# Stage25 Summary RU

classification: `stage25-conv-threading-expand-selected`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `e3bbacba79f2c58b10057735c514a280577223c2`
end_head: `b382bd71c4091cc3476d59f77cb35c2a0d246513`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false

## Proven

Stage24 selected path был воспроизведён: `mismatches=0`, `max_abs_diff=0`, SHA совпадает с ONNX-cut expected output.

Выбран lane C1. Stage25 добавил локальный explicit threaded sidecar для двух Conv в существующем `/model.4` cut runner path:

```text
/model.4/m.0/cv2/conv/Conv
/model.4/cv2/conv/Conv
```

Итоговый selected path:

```text
mean_total_us: 89178.9
stddev_total_us: 268.184
conv_share_pct: 29.3389
activation_share_pct: 36.7808
merge_share_pct: 23.5081
output_quantize_share_pct: 7.37782
```

## Broken

Ничего критичного не сломано: host CTest, cross build, board correctness и FRM sweep прошли.

## Unknown

Полный YOLO26 inference, model FPS, full-image/camera performance и COCO/mAP не проверялись. `/model.4` cut path не является full-model evidence.

## Next

После C1 Conv уже не главный bucket. Следующий шаг: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001`.
