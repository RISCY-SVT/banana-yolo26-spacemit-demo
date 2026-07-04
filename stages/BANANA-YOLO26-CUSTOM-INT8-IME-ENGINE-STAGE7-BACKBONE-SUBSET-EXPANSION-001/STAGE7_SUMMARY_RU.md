# Stage 7 Summary RU

classification: `stage7-backbone-subset-correct-but-activation-dominates`

Stage 7 расширил доказанный Stage 6 subset до границы `candidate_D_block0_silu_model1_silu_model2_cv1_conv`, то есть дошел до `/model.2/cv1/conv/Conv` без включения `/model.2/Split` и без graph-wide scheduler.

Подтверждено:

- ONNX CPU oracle создан для двух детерминированных fixture inputs.
- Host CTest прошел: `19/19`.
- RISC-V cross build прошел.
- Board CPU0/1/2/3 correctness прошел с `mismatches=0` для Conv0, Act0 handoff, Conv1, Act1 handoff и Conv2.
- CPU0 microbench selected-subset: scalar `1.22366e+06 us`, IME `593347 us`, speedup `2.0623x`.

Главный остаточный bottleneck: activation/requant fallback `436780 us`, то есть `73.6129%` от Stage 7 IME total. Поэтому следующий рекомендуемый шаг — отдельный Stage 8 для activation/requant optimization.

Не сделано и не заявлялось: full YOLO26 inference, model FPS, COCO/mAP, camera demo, production readiness, ncnn integration, XSlim, vmadot sliding ops.
