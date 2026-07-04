# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE8-ACTIVATION-REQUANT-OPTIMIZATION-001

Mission: optimize the activation/requant fallback buckets proven dominant in Stage 7 while preserving Stage 7 selected-subset correctness.

Scope:

- target `act0_requant_us` and `act1_requant_us` in `candidate_D_block0_silu_model1_silu_model2_cv1_conv`;
- keep Conv execution on the existing plain `smt.vmadot` MMT4D path;
- implement only local LUT/fixed-point/vector-safe activation/requant alternatives that are oracle-checked;
- compare against ONNX CPU oracle and existing scalar float fallback;
- measure activation-only and full selected-subset timing separately;
- no full YOLO26 engine, no graph scheduler, no COCO/mAP, no camera, no model FPS, no XSlim, no `/data/ncnn` mutation, no vmadot sliding implementation.

Acceptance gate:

- host and board CPU0-3 correctness pass with mismatches 0 for handoff tensors;
- activation/requant total is meaningfully reduced versus Stage 7 `436780 us` CPU0 baseline;
- selected-subset total remains correct and faster than scalar;
- no production claim.
