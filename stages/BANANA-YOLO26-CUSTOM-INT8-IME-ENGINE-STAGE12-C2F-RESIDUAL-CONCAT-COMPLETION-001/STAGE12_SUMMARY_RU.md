# Stage 12 Summary RU

classification: `stage12-c2f-block-complete-ready-for-next-block-stage`

Stage 12 завершил первый `/model.2` C2f-style boundary в узком custom INT8 IME runner:

- добавлен float-domain `/model.2/m.0/Add`;
- добавлен float-domain `/model.2/Concat`;
- добавлен post-Concat Q/DQ;
- добавлен `/model.2/cv2/conv/Conv` через существующий prepacked `smt.vmadot` MMT4D path;
- full engine, scheduler, camera, COCO/mAP и production claims не добавлялись.

Корректность:

- Host CTest: `29/29` pass;
- RISC-V cross build: pass;
- Board CPU0/1/2/3: pass;
- `concat_mismatches=0`;
- `model2_cv2_mismatches=0`.

CPU0 selected-subset microbench:

- Stage 12 IME A2 total: `582039 us`;
- activation share: `15.1699%`;
- conv share: `47.0785%`;
- add+concat share: `15.4979%`;
- pack/layout share: `22.3855%`.

Главный caveat: Add/Concat корректны, но float split/materialization + post-Concat QDQ стали заметным локальным bottleneck. Это не full-model FPS.
