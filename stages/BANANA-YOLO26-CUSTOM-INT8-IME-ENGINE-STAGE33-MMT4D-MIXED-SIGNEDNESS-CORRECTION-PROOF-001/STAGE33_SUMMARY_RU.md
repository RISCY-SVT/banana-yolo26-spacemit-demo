# STAGE33 SUMMARY RU

classification: `stage33-mixed-signedness-correct-but-regresses`

Stage33 проверил узкий mixed-signedness вариант для текущего `/model.4` ONNX-cut пути:

```text
baseline:  smt.vmadot   s8 x s8
candidate: smt.vmadotus u8 x s8
target:    /model.4/cv2/conv/Conv
```

## Доказано

- `smt.vmadotus` собирается named asm route и виден в objdump.
- Host oracle подтверждает алгебру baseline vs mixed-signedness.
- Board CPU0-3 проходит same-input ONNX-cut проверку:

```text
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass
```

## Сломано / не принято

Candidate убрал measured `/model.4/cv2` correction bucket, но ухудшил общий selected-cut runtime:

```text
total_us: 40380.4 -> 40934.1
model4_cv2_correction_us: 1742.83 -> 0
model4_cv2_conv_us: 11852.7 -> 12862.2
selected_cut_total_regression: 1.37%
```

Acceptance gates не пройдены, поэтому mixed-signedness mode не выбран как runtime path. Он остается только explicit diagnostic/local mode.

## Неизвестно

- Можно ли вернуть выгоду mixed signedness после удаления дополнительного copy/compute overhead.
- Будет ли лучше следующий локальный выигрыш в thread/copy overhead или output QuantizeLinear bucket.

## Валидация

```text
host_ctest: pass (42/42)
riscv_cross_build: pass
board_CPU0_3_correctness: pass
stable_benchmark: pass
same_input_onnx_cut: pass
```

## Следующий шаг

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE34-THREAD-COPY-OR-OUTPUT-QUANTIZE-LOCAL-REPAIR-001`

Stage34 должен смотреть не на mixed signedness как выбранный путь, а на локальный thread/copy overhead вокруг `/model.4/cv2/conv/Conv` или output quantize, если same-session bucket map снова покажет его материальным.

Non-claims:

```text
not full YOLO26 inference
not model FPS
not full-image/camera performance
not COCO/mAP
not production/default-backend readiness
```
