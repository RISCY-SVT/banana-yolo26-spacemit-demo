# Stage 9 Final Report

classification: `stage9-rvv-requant-lut-fusion-ready-for-backbone-expansion`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE9-RVV-REQUANT-LUT-FUSION-GATE-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `71e143271b2d09eb35511725e360c3c95bddfc09`
end_head: `pending-local-commit-see-final-response`
pushed: false
full_engine_implemented: false
ncnn_source_mutated: false
production_claim_made: false
xslim_used: false
vmadot_sliding_used: false
selected_mode: `A2_rvv_f32_lut`
activation_share_before: `55.0266%`
activation_share_after: `13.4148%`
selected_subset_total_before_us: `350531`
selected_subset_total_after_us: `182420`
accepted_path: `RVV f32 int32->uint8 requant + 256-entry int8 SiLU LUT`

## Proven

- Stage 8 baseline replay passed.
- Host CTest passed: `25/25`.
- RISC-V cross build passed.
- Board CPU0/1/2/3 correctness passed.
- A2 RVV f32 LUT matched Stage 8 LUT reference and selected-subset Conv2 output with `mismatches=0`.
- ONNX Runtime standalone 256-code Q/DQ+SiLU LUT oracle matched Stage 8 LUT for Act0 and Act1.
- Stage 9 minimum, good, and excellent timing gates passed for the selected subset.

## Broken

- A5 packA handoff is only a sidecar. It is correct but not integrated into the selected runner.
- A1/A4 scalar-unrolled paths improved over A0 but still left activation share above 40%.

## Unknown

- Full YOLO26 inference correctness or speed.
- COCO/mAP impact.
- Full-image/camera behavior.
- Whether A2 remains exact for future graph nodes with different scales/tensor distributions.

## Validation

- `stage8_baseline_replay_report.md`: pass.
- `correctness_report.md`: pass.
- `component_timing_report.md`: A2 selected-subset total `182420 us`, activation share `13.4148%`.

## Next

Recommended next stage after review/approval:

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`
