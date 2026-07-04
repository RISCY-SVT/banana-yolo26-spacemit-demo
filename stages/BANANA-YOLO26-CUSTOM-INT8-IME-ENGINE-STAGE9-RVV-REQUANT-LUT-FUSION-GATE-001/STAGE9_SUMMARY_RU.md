# Stage 9 Summary

classification: `stage9-rvv-requant-lut-fusion-ready-for-backbone-expansion`

Stage 9 уменьшил остаточный activation/requant bottleneck для selected subset `candidate_D_block0_silu_model1_silu_model2_cv1_conv`.

Принятый путь: `A2_rvv_f32_lut`.

Ключевые результаты CPU0:

| metric | before A0 Stage 8 LUT | after A2 RVV |
|---|---:|---:|
| selected_subset_total_us | 350531 | 182420 |
| activation_total_us | 192885 | 24471.3 |
| activation_share | 55.0266% | 13.4148% |
| mismatches | 0 | 0 |

Проверено:

- Host CTest: `25/25` pass.
- RISC-V cross build: pass.
- Board CPU0/1/2/3 correctness: pass.
- ONNX Runtime standalone 256-code LUT oracle: pass for Act0 and Act1.
- `/data/ncnn` не изменялся.
- XSlim не использовался.
- Full engine, COCO/mAP, camera и model FPS claims не делались.

Следующий рекомендуемый шаг после review/approval:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`
