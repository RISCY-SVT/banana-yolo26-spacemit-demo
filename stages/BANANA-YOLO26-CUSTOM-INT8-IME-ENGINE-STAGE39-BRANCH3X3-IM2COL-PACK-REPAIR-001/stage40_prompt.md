# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001

## Mission

Build a minimal, correctness-first full-model runner skeleton/dataflow gate for YOLO26 custom INT8 IME, using the current selected `/model.4` ONNX-cut evidence as one proven block but not claiming model FPS or production readiness.

## Context

Stage39 produced an explicit local `Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK` sidecar. It preserved byte-exact ONNX-cut output and improved selected-cut total, but it did not close the direct branch3x3 im2col/pack speed gate:

```text
selected_cut_total: 30334.500 -> 28073.000 us
selected_cut_total_speedup: 1.080558x
combined_branch3x3_conv: 10291.260 -> 8882.390 us
combined_branch3x3_im2col_pack: 5558.070 -> 5355.820 us
im2col_pack_speedup: 1.037763x, below 1.30x gate
```

## Boundaries

No production/default backend claim, no camera/full-image demo, no COCO/mAP, no graph-wide scheduler, no `/data/ncnn` mutation, no CPU4-7 IME.

## Recommended Gate

1. Preserve the Stage39 selected-cut modes as explicit local modes.
2. Build a minimal full-model runner skeleton that can route tensors through known-good blocks and explicit fallbacks.
3. Prove tensor boundary correctness block-by-block before any performance claim.
4. Decide whether broader memory planning or selected next-block expansion has higher ROI than more `/model.4` micro-tuning.
