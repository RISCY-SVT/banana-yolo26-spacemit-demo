# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE45-MODEL5-6-COMBINED-ISLAND-UPPER-BOUND-AND-ROADMAP-DECISION-001

## Mission

Run a decision-only upper-bound study for a contiguous model5-to-model6 custom island. Do not implement model6 unless a later separately approved stage authorizes it.

## Accepted input

- Stage44 classification: `stage44-model5-exact-no-net-win`.
- Fixed host ORT policy remains correctness authority.
- R2a model5 is exact but not performance-selected.
- Resource-matched board ORT model5 intra4: `11701.121842 us`.
- Final R2a model5 paired mean: `24157.4 us`.
- Final model4-only scaffold: `510864.440692 us`.
- Final model4-to-model5 scaffold: `515063.225590 us`.

## Required decision

1. Measure an isolated resource-matched model6 ORT baseline and graph-valid model5/model6 boundaries.
2. Compute a conservative upper bound for removing the model5/model6 ORT boundary, layout adapters, and duplicate activation/requant work.
3. Account for model6 branch/merge, workspace, weight prepack, and persistent-layout costs.
4. Choose exactly one: authorize a future model5-6 implementation proof, pivot to another board-ranked block, or stop the custom-engine expansion lane.

## Boundaries

No model6 implementation, no new ISA, no CPU4-7 IME, no full engine, no camera/COCO/mAP, no FPS/production/default claim, no push.
