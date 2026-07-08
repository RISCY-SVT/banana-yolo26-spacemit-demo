# Stage33/Stage32 Required Artifact Read Log

The Stage33/Stage32 reports were read before code edits. Equivalent commands:

- `sed -n "1,220p" stages/...STAGE33.../STAGE33_FINAL_REPORT.md`
- `sed -n "1,220p" stages/...STAGE33.../candidate_benchmark_report.md`
- `sed -n "1,220p" stages/...STAGE33.../model4_cv2_signedness_contract.md`
- `sed -n "1,220p" stages/...STAGE33.../bucket_attribution_report.md`
- `sed -n "1,220p" stages/...STAGE33.../conv_roofline_stage33.md`
- `sed -n "1,240p" stages/...STAGE32.../mmt4d_component_replay_report.md`
- `sed -n "1,220p" stages/...STAGE32.../integer_dot_signedness_family_report.md`

Key facts recovered:

- Stage33 candidate `smt.vmadotus u8 x s8` was byte-exact but rejected for performance.
- Stage33 baseline `branch1_add_lut`: total_us=40380.4, model4_cv2_conv_us=11852.7, model4_cv2_compute_us=8129.4, model4_cv2_correction_us=1742.83.
- Stage33 mixed candidate: total_us=40934.1, model4_cv2_conv_us=12862.2, model4_cv2_compute_us=9699.05, model4_cv2_correction_us=0, model4_cv2_copy_us=1127.18.
- Stage32 per-conv MMT4D showed  compute was the largest raw sub-bucket.
- Stage32 signedness family proved `smt.vmadot`, `smt.vmadotu`, `smt.vmadotsu`, `smt.vmadotus`; Stage34 must not continue `vmadotus` as selected path.
