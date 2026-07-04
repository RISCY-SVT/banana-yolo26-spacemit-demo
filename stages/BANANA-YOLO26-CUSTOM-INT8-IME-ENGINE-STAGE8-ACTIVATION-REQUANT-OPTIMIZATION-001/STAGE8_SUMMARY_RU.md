# Stage 8 Summary RU

classification: `stage8-activation-improved-but-still-dominates`

Stage 8 не расширял граф и не делал full YOLO26 inference. Работа была ограничена selected subset `candidate_D_block0_silu_model1_silu_model2_cv1_conv` из Stage 7.

Сделано:

- добавлен `int8_lut` activation/requant path для Act0 и Act1;
- добавлены 256-entry SiLU LUT для каждой границы;
- добавлен диагностический `fixed_requant_only`;
- текущий float fallback сохранён как `scalar_float_reference`;
- добавлены host tests и board correctness для CPU0/1/2/3.

Результат CPU0 microbench:

| metric | before | after |
|---|---:|---:|
| selected-subset IME total | `620735 us` | `350092 us` |
| activation/requant total | `465901 us` | `192568 us` |
| activation share | `~75.06%` | `55.0052%` |

Корректность:

- Host CTest: `21/21` pass.
- RISC-V cross build: pass.
- Board CPU0/1/2/3 correctness: pass.
- LUT exhaustive oracle: pass.
- `mismatches=0` для selected subset.

Остаточный bottleneck: activation/requant всё ещё занимает больше 40% total. Поэтому следующий шаг должен быть не расширением графа, а `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE9-ACTIVATION-FUSION-AND-PACK-HANDOFF-001`.

Не делалось и не заявлялось: full YOLO26 inference, graph-wide scheduler, camera, COCO/mAP, model FPS, production readiness, `/data/ncnn` mutation, XSlim, `vmadot1/2/3`, `vmadotn`, FP/vfmadot.
