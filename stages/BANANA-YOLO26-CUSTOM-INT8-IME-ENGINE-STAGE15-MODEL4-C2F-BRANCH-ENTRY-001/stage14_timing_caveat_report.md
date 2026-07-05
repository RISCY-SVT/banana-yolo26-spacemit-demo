# Stage 14 Timing Caveat Report

Stage 14 reported `total_us=139.04` for `stage14_IME_A2_rvv_f32_lut`.

That number is compact selected-subset evidence only:

- It uses compact deterministic fixtures.
- It is not full-shape `/model.3` or `/model.4` performance.
- It is not full YOLO26 inference.
- It is not full-image speed.
- It is not camera speed.
- It is not COCO/mAP or production evidence.

Stage 15 must not compare the compact Stage 14 `139.04 us` directly against earlier full-shape selected-subset timings from Stage 12/13.

Stage 15 timing policy:

- Report compact correctness fixture timing separately.
- Prefer representative/full-shape selected-subset timing when feasible.
- If representative/full-shape timing is not feasible in Stage 15, record `full_shape_stage15_timing: not_proven`.

Traceability:

- Stage 14 tracked report placeholders were patched to the actual Stage 14 commit `5cc09059f83eaef6af8c9a6aee3eab1e4edd46e7`.
- No Stage 14 numeric results or scientific conclusions were changed.
