# Stage 50: Persistent NCHWc8 Model4-Final-to-Model8 LUT-v2 Closure Gate

## Baseline

- Start from the Stage49 local commit on `yolo26-custom-int8-engine`.
- Preserve `K1X_INT8_V1`, manifest `0d3c3d49abdc8dd83857af223ea63bcb7a31058be4bcdb7cd7e6ccdf35659bac`, the exact model4-final-to-model6 regression, CPU0-3 IME, controller-only CPU4, and the frozen RT205 lane.
- Accepted Stage49 internal slice: 26710.414338 us mean versus B120 ORT 42036.659040 us.

## Mission

1. Add stable exact LUT-v2 rows for 1x1 NCHWc8, 3x3 stride1, N4/N8/N16 output classes, Add, and Concat placement.
2. Extend the same deterministic package and resident arena from model6 through model8, with F0-F7 Python/C++/board scalar/IME exactness at every integer boundary.
3. Measure model4-final-to-model8 internal custom and equivalent B120 ORT cuts under 10/100/5.
4. Update measured current-graph coverage without claiming a full executor or student performance.

## Hard limits

No RT205 work, model training, student selection, CPU4-7 IME, new ISA/opcode, default dispatch, full graph executor, camera/COCO claim, push, or `/data/ncnn` mutation.

## Decision

Classify whether the widened resident slice remains at least competitive within +/-10% of resource-matched ORT. If negative, stop current-graph expansion and prepare an evidence-backed student architecture decision. If positive, authorize only the next bounded shape-coverage/full-graph-estimator stage.
