# STAGE39 SUMMARY RU

classification: stage39-im2col-pack-partial-total-win-im2col-gate-missed

## Broken / Proven / Unknown

Proven:
- Новый explicit mode `Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK` сохраняет same-input ONNX-cut byte equality: `mismatches=0`, `max_abs_diff=0`, SHA `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`.
- FRM sweep `RNE/RTZ/RDN/RUP/RMM` проходит.
- В same-session board benchmark selected-cut total улучшился `1.080558x`: `30334.500 us -> 28073.000 us`.
- Combined branch 3x3 conv total улучшился `1.158614x`: `10291.260 us -> 8882.390 us`.

Broken:
- Главный Stage39 im2col/pack gate не закрыт: combined branch3x3 im2col_pack speedup только `1.037763x`, требовалось `>=1.30x`.

Unknown:
- Точная доля gather vs packA внутри fused A-panel path не отделена без более intrusive inner-loop timing.
- Не доказано, что дальнейший selected-cut im2col micro-tuning даст >=5% total improvement.

## Что изменилось

- Добавлен explicit local sidecar mode `Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK`.
- Добавлен fast 3x3 A-panel pack path: 8-channel chunk copy/fill для поддерживаемых full panels, edge fallback сохраняет старую семантику.
- Stage37 baseline mode сохранён для A/B replay.

## Не утверждалось

Это не full YOLO26 inference, не model FPS, не full-image/camera performance, не COCO/mAP и не production/default-backend readiness.

## Следующий шаг

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001`: full-model runner skeleton / broader dataflow gate вместо дальнейшей узкой selected-cut im2col micro-tuning.
