# Stage43 Prompt

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE43-CONTIGUOUS-MODEL5-8-ISLAND-ORACLE-AND-FIRST-BLOCK-GATE-001`

## Mission

Extend the fixed-host-oracle and in-process C++ scaffold from the current custom model4 island toward the adjacent `model.5 -> model.6 -> model.7 -> model.8` region. Build graph-verified quantized contracts and isolated board profiles, then implement at most one bounded first block if its byte-exact oracle and ROI gates pass.

## Required starting facts

- Stage42 reference policy: fixed host ORT 1.27.0, operational `ORT_ENABLE_ALL` contract.
- Board ORT 1.20.2+spacemit is fallback/integration/timing only.
- Same-input board scalar and IME model4 outputs are byte-exact against the host oracle.
- Corrected scaffold: prefix `229662.287042 us`, custom model4 `25149.098496 us`, adapters `17131.705674 us`, suffix `554052.102166 us`.
- Model16 semantic/quantized oracle package exists but is a non-contiguous reuse side lane.

## Scope

1. Generate fixed-host semantic and quantized boundary oracles for model5, model6, model7, and model8.
2. Measure each block directly on board with one in-process ORT session per isolated block; do not infer isolated costs by subtracting cumulative sessions.
3. Rank by measured board cost, clean Q/DQ boundary, kernel reuse, and layout churn.
4. Implement at most one selected block, likely model5 or model6, through an explicit non-default mode.
5. Require exact uint8 boundary equality against the fixed host oracle, CPU0-3-only IME, FRM restoration, host tests, cross build, and stable board timing.

## Exclusions

No model9 SPPF, model10 attention, model23 head, full engine claim, model FPS, camera, COCO/mAP, new ISA lane, `/data/ncnn` mutation, CPU4-7 IME, default dispatch, or push.
