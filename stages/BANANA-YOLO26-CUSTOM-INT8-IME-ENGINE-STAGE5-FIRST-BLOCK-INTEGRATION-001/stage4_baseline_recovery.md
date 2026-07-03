# Stage 4 Baseline Recovery

Reference stage: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001`

Recovered from:

- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/STAGE4_FINAL_REPORT.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/microbench_stage4_report.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/conv1x1_stage4_report.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE4-PACKING-REPAIR-001/conv3x3_stage4_report.md`

## Recovered Facts

- Stage 4 classification: `stage4-packing-repaired-ready-for-first-block-integration`
- Implementation primitive: plain `smt.vmadot` MMT4D `4x4x8 s8xs8->s32`
- No `vmadot1/2/3`, `vmadotn`, FP/vfmadot, full YOLO26 engine, graph scheduler, camera, COCO/mAP, or model-level FPS work was done.
- Host-native Stage 4 CTest: `16/16` pass.
- RISC-V cross build with `Y26_K1X_ENABLE_IME=ON`: pass.
- Board cluster0 Stage 4 tests: pass.

## Stage 4 Board Baseline

| case | scalar us | old wrapper us | Stage 3 prepacked us | Stage 4 M-major us | Stage 4 N-major us | packA probe us | correction us |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conv1x1 `160x160x32->32` | `112047` | `137374` | `46649.6` | `21843.2` | `65855.4` | `6260.54` | `2649.96` |
| Conv3x3 `160x160x16->8` | `147333` | `391010` | `149121` | `37097.9` | `65852.3` | `24951.5` | `658.726` |

Stage 5 starts from commit `55f133ca0900cb91d891a77149479b6fd392c420`.
