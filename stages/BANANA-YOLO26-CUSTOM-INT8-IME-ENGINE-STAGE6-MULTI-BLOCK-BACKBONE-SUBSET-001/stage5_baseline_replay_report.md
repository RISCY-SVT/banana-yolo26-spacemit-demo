# Stage 5 Baseline Replay Report

Source reports:

- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001/STAGE5_FINAL_REPORT.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001/block_correctness_report.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE5-FIRST-BLOCK-INTEGRATION-001/block_microbench_report.md`

Recovered facts:

- Stage 5 classification: `stage5-first-block-ready-for-multiblock-stage`
- Selected block: `block0_conv_only`
- Node: `/model.0/conv/Conv`
- Shape: `640x640x3 -> 320x320x16`
- Kernel: `3x3`, stride `2`, padding `1`
- Board CPU0/1/2/3 correctness: pass
- Board microbench CPU0:
  - scalar total: `463480 us`
  - IME total packing included: `71932.7 us`
- Downstream SiLU and requant were not implemented in Stage 5.
- XSlim was not used and remains unauthorized.
- No full YOLO26 inference, model FPS, COCO/mAP, camera, ncnn mutation, or production claim was made.

Stage 6 will replay the comparable Conv0 baseline through the same Stage 5 runner in host/board validation and will separately report incremental activation and Conv1 costs.

