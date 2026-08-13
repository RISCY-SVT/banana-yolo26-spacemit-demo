# XSLIM-DEV-001 Charter

This is a design charter, not implementation authorization.

## Phase 1: generic causal localizer

Implement `xslim-localize-quant-error` with safe complete cut sets, exact source/QDQ correspondence, bidirectional split/splice reconstruction controls, a task-evaluator adapter, paired image-level bootstrap, and bounded `custom_setting` proposal output. It must fail closed on incomplete cuts, ambiguous mappings, dynamic shapes, shared unsafe Q/DQ, or reconstruction mismatch.

## Phase 2: target-runtime profiles

### SpacemiT profile

Enforce signed S8-QDQ only, zero QLinear, zero UINT8 zero points, explicit Conv `kernel_shape`, the six-output/unquantized-tail contract, and a mixed-precision/fallback risk report. Expected fused regions are hypotheses, never placement or speed claims.

### K1X custom-engine profile

Design a new `K1X_INT8_V2` contract: per-op scales/zero points, per-channel weight scales, exact multiplier/right shift, ties-to-even rounding, saturation, accumulator bounds, qdomain graph, residual/concat alignment, NCHWc8 hints, and packed-weight manifest. Do not mutate or reinterpret `K1X_INT8_V1`.

## Phase 3: task-aware robust selection

Add a YOLO COCO metric plugin, AP-small/medium/large constraints, multiple deterministic calibration draws/seeds, a variance penalty, and score-collapse/non-finite guards.

## Phase 4: targeted generation

Generate at most two evidence-selected policies deterministically, then require H500/full COCO and signed-QDQ conformance. This phase needs separate authorization.

## Phase 5: hardware feedback

Measure SpacemiT provider partitions, CPU fallback/conversion cost, custom-engine rescale/layout cost, matched board latency, correctness, and soak. A target profile alone proves none of these hardware properties.
