# Stage27 Summary RU

classification: `stage27-conv-decision-selected-tile-prepack-future-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `6a32c904cf711afab24e3efd9d2adaa9306c101f`
end_head: `see-final-head-copy-after-local-commit`
pushed: false

## Доказано

- Stage26 accepted mode `Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT` воспроизведён через real runner API.
- Same-input ONNX-cut correctness сохранилась: `mismatches=0`, `max_abs_diff=0`, SHA совпал.
- FRM sweep прошёл для `RNE/RTZ/RDN/RUP/RMM`, post-call FRM восстановлен.
- Stable timing protocol выполнен: `taskset -c 0-3`, `warmup=10`, `runs=100`, `repeats=5`.
- Conv остаётся dominant bucket: `26869.6 us`, `64.4832%`.
- Все текущие Conv nodes уже выигрывают от 4-thread cluster0 policy; сниженные thread counts ухудшали total.

## Сломано

- Новых correctness-регрессий нет.
- Локальный Stage27 repair не принят: current per-node threaded workers уже persistent, thread-count threshold не улучшил timing, output quantize ниже 20%.

## Неизвестно

- Точный split внутри Conv bucket: pack/im2col/correction/compute по каждому node ещё не доказан.
- Лучший MMT4D tile/prepack/correction candidate неизвестен.
- `vmadot1/2/3` не проверялся и не реализовывался.
- Full YOLO26 FPS, full-image/camera, COCO/mAP и production readiness неизвестны.

## Следующий шаг

Рекомендуемый Stage28:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001
```

Track B отдельно:

```text
BANANA-YOLO26-VENDOR-ORT-RT204-MAP-BASELINE-001
```
