# Stage 14 Summary RU

classification: `stage14-next-c2f-expanded-ready-for-next-stage`

Stage 14 расширил покрытие после `/model.2` до bounded subset:
`candidate_H3_model2_act_model3_act_model4_cv1_conv`.

Что доказано:

- Stage 13 replay прошел без регрессии.
- Stage 13 traceability исправлена на `9219f897a47d76e8b06031d29dcc18c498cf48a0`.
- Новый subset остановлен до `/model.4/Split`, без нового Add/Concat и без graph-wide scheduler.
- ONNX Runtime LUT oracle для новых activation/QDQ boundaries прошел с mismatches `0`.
- Host CTest: `31/31`.
- RISC-V cross build: pass.
- Board CPU0/1/2/3 correctness: pass.
- CPU0 compact selected-subset microbench для `stage14_IME_A2_rvv_f32_lut`: `total_us=139.04`, `conv_share_pct=69.1314`, `activation_share_pct=16.0846`, `merge_share_pct=8.54121`, `pack_layout_share_pct=0.339711`.

Не делалось:

- full YOLO26 engine;
- graph-wide scheduler;
- camera/full-image demo;
- COCO/mAP;
- production/model FPS claim;
- ncnn mutation;
- XSlim;
- `vmadot1/2/3`, `vmadotn`, FP/vfmadot.

Следующий рекомендуемый шаг:
`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE15-MODEL4-C2F-BRANCH-ENTRY-001`.
