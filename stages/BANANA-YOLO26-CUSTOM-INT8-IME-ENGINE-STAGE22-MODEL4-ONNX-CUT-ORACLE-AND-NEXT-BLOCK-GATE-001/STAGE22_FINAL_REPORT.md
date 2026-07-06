# STAGE22_FINAL_REPORT

classification: `stage22-onnx-cut-pass-ready-for-next-repair`
stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`
repo: `/data/banana-yolo26-spacemit-demo`
branch: `yolo26-custom-int8-engine`
start_head: `d8025985bff6373aaf7082a47ad532a18bd64134`
end_head: `pending-local-commit-see-final-response`
pushed: `false`
full_engine_implemented: `false`
ncnn_source_mutated: `false`
production_claim_made: `false`

## Summary

Stage22 constructed a true same-input ONNX Runtime CPU cut for the `/model.4` C2f boundary and compared the integrated C++ selected path against that cut on the exact same full-shape boundary tensor.

```text
onnx_cut_status: pass
engine_vs_onnx_mismatches: 0
engine_vs_onnx_max_abs_diff: 0
rounding_regression_status: pass
host_tests: pass
cross_build: pass
board_tests: pass
stable_timing_mean_total_us: 225214
stable_timing_stddev_total_us: 44.6982
stable_timing_cv_pct: 0.019847
```

This is not full YOLO26 inference, not full-image/camera performance, not COCO/mAP, not model FPS, and not production/default-backend evidence.

## Proven

- ONNX cut construction succeeded for `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output` to `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`.
- ONNX cut output matches full-model ORT output for the same synthetic seeded image at the cut boundary.
- Host C++ scalar same-input path matches ONNX cut output exactly.
- Board `ime_threaded` same-input path matches ONNX cut output exactly.
- Ambient `frm` sweep RNE/RTZ/RDN/RUP/RMM passes with `mismatches=0` and restores `frm`.
- Host CTest passed `36/36`; RISC-V cross build passed.

## Broken

- Before Stage22 repair, ambient `RTZ` caused one mismatch in the same-input verifier. Stage22 fixed this with scoped RNE control around the selected verifier invocation.

## Unknown

- Full YOLO26 model correctness and mAP remain unknown.
- Full-image/camera performance remains unknown.
- The Stage22 same-input cut runner has adapter/verification overhead not fully attributed to named buckets.

## Validation Status

```text
git_diff_check: pass
host_ctest: pass
cross_build: pass
board_same_input_cut: pass
rounding_regression: pass
result_packet: pending-export
```

## Next

`BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`
