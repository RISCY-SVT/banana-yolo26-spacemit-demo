# Stage40 Replay Report

classification: stage40-replay-pass-for-all-ORT-host-oracle

Stage41 regenerated the Stage40 full-model ORT CPU reference and block cuts from the accepted model:

```text
model: .deps/models/yolo26/int8_rt204_forensics/manual_ort/manual_e2e_rep_conv_matmul_qdq.onnx
model_sha256: 30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c
input: images float32 1x3x640x640
output: output0 float32 1x300x6
input_mode: synthetic_seeded
```

Regenerated cuts:

```text
prefix_images_to_model4_input: 73 nodes
model4_input_to_model4_output: 31 nodes
suffix_model4_output_to_output0: 966 nodes
```

All-ORT cut replay matched the full ORT CPU output exactly:

```text
prefix_vs_full_model4_input: mismatches=0 max_abs_diff=0
all_ort_model4_vs_full_model4_output: mismatches=0 max_abs_diff=0
all_ort_final_vs_full_reference: mismatches=0 max_abs_diff=0
```

Stage40 Python/file timings are retained only as skeleton profiling, not model FPS:

```text
full_ort_reference: 197743.799 us
prefix_images_to_model4_input: 59987.179 us
model4_cut_all_ort: 9756.942 us
suffix_model4_output_to_output0: 129679.221 us
```
