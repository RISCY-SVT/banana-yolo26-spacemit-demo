# Stage 6 Summary RU

classification: `stage6-multiblock-ready-for-backbone-subset-stage`

Stage 6 расширил Stage 5 с одного Conv-only блока до bounded subset:

```text
/model.0/conv/Conv
Conv0 Q/DQ
SiLU: /model.0/act/Sigmoid + /model.0/act/Mul
Act0 Q/DQ
/model.1/conv/Conv
```

Выход Stage 6: corrected int32 output для `/model.1/conv/Conv`. Полный YOLO26 engine, graph scheduler, camera, COCO/mAP, model FPS и production claims не делались.

Доказано:

- ONNX CPU oracle: pass.
- Host CTest: `18/18` pass.
- RISC-V cross build: pass.
- Board CPU0/1/2/3 correctness: pass, mismatches `0`.
- Board CPU0 microbench для выбранного subset: scalar `1009980 us`, IME `419769 us`.
- IME быстрее scalar для выбранного subset примерно `2.41x`.

Главное ограничение: activation/requant fallback пока scalar float и занимает `286942 us` из `419769 us` IME total. Следующий этап должен аккуратно расширять backbone subset и отдельно решать стоимость activation/requant.

`smt.vmadot` plain MMT4D остаётся единственной IME primitive. XSlim, `vmadot1/2/3`, `vmadotn`, FP/vfmadot не использовались. `/data/ncnn` и YOLO11 production repo не изменялись.

Следующий рекомендуемый шаг после review/approval:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE7-BACKBONE-SUBSET-EXPANSION-001
```

