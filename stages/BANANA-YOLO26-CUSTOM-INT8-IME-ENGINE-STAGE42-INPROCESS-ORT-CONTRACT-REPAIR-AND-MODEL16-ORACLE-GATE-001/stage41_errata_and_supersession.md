# Stage41 Errata And Supersession

Stage42 preserves all Stage41 raw evidence. The following statements supersede interpretation only; historical files are not rewritten.

## Attempt History

Stage42 attempt 1 was an administrative approval check only. It performed no repository inspection, build, board execution, source change, commit, push, or technical result. This direct-user-authorized rerun supersedes that administrative stop.

## Tensor Contract

The accepted graph metadata was regenerated from the fixed model and confirms:

- input: `/model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output`, uint8, NCHW `1x64x80x80`; custom layout NHWC `1x80x80x64`.
- output: `/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output`, uint8, NCHW `1x128x80x80`; custom layout NHWC `1x80x80x128`.

Stage42 publishes these graph-verified values in `boundary_tensor_manifest.tsv`.

## Timed-Loop Contamination

Stage41 `run_pipeline_protocol` executed the ORT model4 reference session inside every warmup and timed custom-pipeline iteration. Its wall time was included in `total_us` but omitted from the named component sum. Consequently Stage41 total/attribution values are skeleton diagnostics, not accepted custom-pipeline timing.

Stage42 separates:

- validation: full ORT, model4 ORT, custom model4, comparisons and dumps;
- benchmark: prefix, layout-in, custom model4, layout-out, suffix only.

## Cumulative Suffix Profile

Stage41 block deltas were differences between independently built cumulative ORT sessions. Negative deltas and high CV demonstrate that these are ranking diagnostics, not isolated block costs. Stage42 does not accept model16 solely from that subtraction.

## Provenance And Capture

Stage41 central command logging was incomplete and its board suffix profile was truncated near model22. Stage42 uses a TSV command ledger with separate stdout/stderr files. It preserved the complete model.0 through output0 boundary matrix and did not repeat the truncation.
