# Stage36 Summary RU

classification: stage36-cv2-pipelined-vmadot-selected
stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE36-CV2-PIPELINED-VMADOT-CANDIDATE-001
start_head: a8b76072f19ff792bc5afc33ab93a022f2c26eb6
end_head: pending-local-commit-see-final-response
pushed: false

## Что сделано

Stage36 добавил явный локальный режим для выбранного `/model.4` ONNX-cut пути:

- `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4`
- `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED6`

Принят кандидат `A1_branch1_add_lut_cv2_pipelined4`: 4 независимые accumulator-группы для signed-storage `s8 x s8 -> s32` `smt.vmadot` в реальном `/model.4/cv2/conv/Conv` 1x1 Conv. `smt.vmadotus`, `vmadot1/2/3` integration и `vmadotn` не использовались.

## Proven

- Same-input ONNX-cut byte equality сохранена: `mismatches=0`, `max_abs_diff=0`.
- Output SHA совпадает: `70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433`.
- FRM sweep прошел для `RNE/RTZ/RDN/RUP/RMM`.
- Board CPU0 single-thread correctness прошел.
- Board CPU0-3 threaded correctness и stable benchmark прошли.
- Host CTest прошел: 42/42.
- RISC-V cross build с `Y26_K1X_ENABLE_IME=ON` прошел.
- IME не выполнялся на CPU4-7.

## Performance

Stable protocol: `taskset -c 0-3`, `warmup=10`, `runs=100`, `repeats=5`.

| mode | total_us mean/stddev | model4_cv2_compute_us | model4_cv2_conv_us | status |
| --- | ---: | ---: | ---: | --- |
| A0 baseline | 37341.1 / 405.161 | 7541.75 | 10420.4 | pass |
| A1 pipelined4 | 33192.7 / 364.104 | 3616.14 | 6307.54 | selected |
| A2 pipelined6 | 33217.5 / 418.787 | 3822.91 | 6521.77 | correct, not selected |

A1 speedups:

- `model4_cv2_compute_us`: 2.085580x
- selected-cut `total_us`: 1.124979x
- `model4_cv2_conv_us`: 1.652055x

## Broken

- A2 6-accumulator candidate не сломан по correctness, но не быстрее A1 в этом session.
- Полный YOLO26 engine не реализован и не проверялся.

## Unknown

- Full-model FPS неизвестен.
- Full-image/camera performance неизвестна.
- COCO/mAP не запускался.
- Производственная готовность и default backend readiness не заявляются.

## Next

Рекомендуемый Stage37:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001`

Следующий этап должен использовать A1 как baseline, заново снять bucket attribution и выбрать одну локальную lane: branch 3x3 Conv/thread-overhead repair, output QuantizeLinear repair, либо остановиться, если локальная selected-cut оптимизация больше не дает убедительного выигрыша.
