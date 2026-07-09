# STAGE38 SUMMARY RU

## Broken

- Нет полного YOLO26 inference.
- Нет model FPS, full-image/camera performance, COCO/mAP или production/default-backend claim.
- Branch 3x3 `im2col_pack` остается крупным локальным bucket после ремонта output QuantizeLinear.

## Proven

- Stage37 selected mode replay прошел same-input ONNX-cut gate: `mismatches=0`, `max_abs_diff=0`, SHA совпал.
- FRM sweep `RNE/RTZ/RDN/RUP/RMM` прошел для replay и Stage38 candidate.
- Новый explicit output quantize mode `Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE` byte-exact.
- Lane A принят:
  - `output_quantize_us: 7055.2 -> 4551.97`
  - speedup output QuantizeLinear: `1.54994x`
  - selected-cut total: `32890.5 -> 30341.5 us`
  - selected-cut speedup: `1.08401x`

## Unknown

- Улучшится ли следующий локальный bucket: branch 3x3 fused im2col/pack.
- Переносится ли selected-cut оптимизация на будущий broader runner без нового proof.

## What Changed

- Добавлен optional `--measure-im2col-pack` для selected-cut bench.
- Добавлены per-node `im2col_pack_us` timing fields.
- Добавлен explicit local output quantize mode `rvv_direct`.
- Default/backend policy не изменялась.

## Human Decision Needed

Следующий рекомендуемый шаг: Stage39 `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001`.
