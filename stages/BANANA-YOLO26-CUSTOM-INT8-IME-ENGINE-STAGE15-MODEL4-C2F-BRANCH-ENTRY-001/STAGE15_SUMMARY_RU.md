# Stage 15 Summary RU

classification: `stage15-model4-branch-correct-but-fullshape-unproven`

Stage 15 расширил покрытие до `candidate_I_model4_split_first_branch`:

`/model.4/cv1/conv/Conv` corrected int32 -> `/model.4/cv1/act` -> `/model.4/Split` -> `/model.4/Split_output_1` Q/DQ -> `/model.4/m.0/cv1/conv/Conv` -> `/model.4/m.0/cv1/act` Q/DQ.

Доказано:

- Stage 14 replay прошел на host и board CPU0/1/2/3.
- Новые LUT-oracle для `/model.4/cv1 -> Split_output_1` и `/model.4/m.0/cv1 -> act` прошли с `mismatches=0`.
- Host CTest прошел: `32/32`.
- RISC-V cross build прошел.
- Board CPU0/1/2/3 correctness прошел, `mismatches=0`.
- CPU0 compact microbench прошел.

Важное ограничение:

Stage 14 `139.04 us` и Stage 15 `160.038 us` являются compact selected-subset timing. Это не full-shape timing, не full-model FPS, не camera/full-image speed и не production evidence.

Stage 15 не реализует full YOLO26 engine, graph-wide scheduler, ncnn integration, XSlim, `vmadot1/2/3`, `vmadotn`, FP/vfmadot или default multithreading.

Рекомендуемый следующий этап:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE16-MODEL4-C2F-COMPLETION-AND-FULLSHAPE-GATE-001 after review/approval`
