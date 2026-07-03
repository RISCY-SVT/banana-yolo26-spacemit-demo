# Stage 5 Summary RU

classification: `stage5-first-block-ready-for-multiblock-stage`

Stage 5 завершен как узкая интеграция первого блока, без full YOLO26 engine, без camera/full-image demo, без COCO/mAP, без model FPS и без production claims.

Выбран блок `block0_conv_only`: `/model.0/conv/Conv`, форма `640x640x3 -> 320x320x16`, `3x3 stride2 pad1`.

Доказано:

- ONNX CPU oracle создан через isolated `.deps/custom_int8_engine/venv-stage5-onnx`.
- Host CTest: `17/17` pass.
- RISC-V cross build: pass.
- Board CPU0/1/2/3 correctness: pass, mismatches `0`.
- Board block microbench: scalar `463480 us`, IME total packing included `71932.7 us`.
- `smt.vmadot` остается единственной IME primitive; `vmadot1/2/3`, `vmadotn`, FP/vfmadot не использовались.
- XSlim не использовался.

Осталось неизвестным:

- full YOLO26 inference speed;
- full-image pipeline speed;
- COCO/mAP;
- downstream SiLU/requant behavior after block expansion.

Рекомендуемый следующий этап:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE6-MULTI-BLOCK-BACKBONE-SUBSET-001`
