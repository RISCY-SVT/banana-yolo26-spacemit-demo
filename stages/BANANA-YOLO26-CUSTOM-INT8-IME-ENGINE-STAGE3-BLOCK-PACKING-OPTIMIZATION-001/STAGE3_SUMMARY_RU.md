# Stage 3 Summary

classification: `stage3-conv-real-node-correct-but-packing-dominates`

Stage 3 выбрал реальные Conv узлы из accepted manual Q/DQ ONNX artifact, поднял ONNX Runtime CPU oracle через отдельный stage-local venv, добавил prepacked weight path и workspace-backed A-panel packing для `smt.vmadot` MMT4D, а также реализовал correction boundary для выбранных real nodes.

## Доказано

- `xslim` не использовался.
- Host CTest: `15/15` pass.
- Cross build с `Y26_K1X_ENABLE_IME=ON`: pass.
- Board cluster0 smoke: runtime probe, `smt.vmadot`, synthetic Conv1x1/Conv3x3: pass.
- Real Conv1x1 fixture: scalar and IME mismatches `0`.
- Real Conv3x3 fixture: scalar and IME mismatches `0`.
- Conv1x1 prepacked IME быстрее scalar на real-node shape.

## Сломано / ограничено

- Conv3x3 packing/im2col всё ещё доминирует: prepacked IME примерно на уровне или медленнее scalar для выбранной формы.
- Full YOLO26 engine, scheduler, full graph requant, COCO/mAP, camera and production path не реализованы.

## Следующий шаг

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001`
