# Stage 10 Summary RU

classification: `stage10-backbone-expanded-ready-for-branch-stage`

Stage 10 расширил выбранный backbone subset после Stage 9 activation gate. Добавлены Conv2 activation/requant boundary, `/model.2/Split`, compact handoff для `/model.2/Split_output_1` и первый branch Conv `/model.2/m.0/cv1/conv/Conv`.

Полный движок YOLO26 не реализован. Graph-wide scheduler, camera/full-image demo, COCO/mAP, model FPS claims, XSlim, ncnn mutation и sliding vmadot не использовались.

Доказано:

- Stage 9 A2 baseline воспроизведён: `182491 us`, activation share `15.0594%`, mismatches `0`.
- RVV A2 переведён на explicit RNE conversion; board regression с ambient `frm` RNE/RTZ/RDN/RUP/RMM дал `mismatches=0`.
- Новый boundary Conv2 -> Split_output_1 LUT совпал с ONNX Runtime 256-code oracle: `mismatches=0`.
- Host CTest: `27/27 passed`.
- Board CPU0/1/2/3 correctness: `pass`, all mismatches `0`.
- Stage 10 expanded subset A2: `234341 us`, activation share `15.379%`, pack/layout share `0.482372%`.

Новый bottleneck: `conv / IME`, особенно добавленный branch Conv. Рекомендуемый следующий шаг: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE11-BRANCH-BLOCK-EXPANSION-001` после review/approval.
