# Stage40 Hypotheses

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE40-FULL-MODEL-RUNNER-SKELETON-GATE-001

H1: A full-model skeleton with ONNX Runtime CPU fallback/cuts for unimplemented sections can produce deterministic YOLO26 output matching ONNX Runtime CPU reference at explicitly chosen boundaries.

H2: The proven custom `/model.4` cut can be inserted into the skeleton without breaking same-input output equivalence at its output boundary.

H3: Full-model block profiling will identify the next high-value blocks; those blocks may differ from `/model.4`.

H4: If full-model skeleton correctness cannot be closed in this stage, a partial block-chain skeleton with a precise blocker is more valuable than another `/model.4` micro-optimization.

Non-claims:

- This is not full YOLO26 production inference.
- This is not final model FPS.
- This is not full-image/camera performance.
- This is not COCO/mAP.
- This is not production/default-backend readiness.
