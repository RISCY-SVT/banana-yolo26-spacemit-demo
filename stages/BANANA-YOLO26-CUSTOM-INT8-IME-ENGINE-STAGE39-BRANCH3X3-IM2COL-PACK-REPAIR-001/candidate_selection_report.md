# Candidate Selection Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE39-BRANCH3X3-IM2COL-PACK-REPAIR-001
repo: /data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
start_head: 11675ccfbdf905bef92b5fd69f75d08a541a549c


## Selected Candidate

- selected_lane: branch 3x3 im2col/pack dataflow
- selected_candidate: `A1_A2_fast_chunks`
- selected_mode: `Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK + Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE`

The candidate replaces the generic 3x3 A-panel scalar store path for supported full panels with an 8-channel chunk copy/fill path and keeps an edge fallback. It does not change MMT4D compute, correction, output quantization, graph coverage, or default backend behavior.

## Gate Result

| gate | result |
|---|---|
| same-input ONNX-cut correctness | pass |
| FRM sweep | pass |
| combined_branch3x3_im2col_pack speedup >= 1.30x | fail: 1.037763x |
| combined_branch3x3_conv_total speedup >= 1.15x | pass: 1.158614x |
| selected_cut_total speedup >= 1.05x | pass: 1.080558x |

Classification is partial: the selected-cut speed win is real in A/B/A and no-instrument timing, but the direct im2col/pack gate was not met. The mode is retained as an explicit local sidecar, not a default backend.
