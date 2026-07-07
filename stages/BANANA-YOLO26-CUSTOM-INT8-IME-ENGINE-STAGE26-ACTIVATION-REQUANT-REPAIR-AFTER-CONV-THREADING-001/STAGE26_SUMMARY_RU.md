# Stage26 Summary RU

classification: `stage26-activation-requant-repaired-ready-for-next-stage`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE26-ACTIVATION-REQUANT-REPAIR-AFTER-CONV-THREADING-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `b382bd71c4091cc3476d59f77cb35c2a0d246513`
end_head: `pending-local-commit-see-final-response`
pushed: false

## Доказано

- Stage25 selected path был воспроизведён на board с `mismatches=0`, `max_abs_diff=0`, `frm_sweep=pass`.
- Activation/requant bucket был почти полностью branch1 activation: `31536.9 us` из `32790.8 us`.
- Новый локальный режим `Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT` сохранил same-input ONNX-cut bytes: SHA совпал с expected.
- Activation/requant снизился с `32790.8 us` до `3004.46 us`; selected cut total снизился с `90086.8 us` до `41573.9 us`.

## Сломано

- Новых известных correctness-регрессий нет.
- Полный YOLO26 engine, full-image/camera, COCO/mAP и production/default backend не реализовывались.

## Неизвестно

- Full-model FPS и качество модели неизвестны.
- Следующий dominant bucket после ремонта — Conv (`64.3721%`), но Stage26 не выполнял Conv/tile/vmadot123 реализацию.

## Следующий шаг

Рекомендуемый Stage27: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001`.
