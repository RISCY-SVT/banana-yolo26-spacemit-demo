# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

## Mission

Continue in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine`.

Stage24 repaired the `/model.4` ONNX-cut merge/post-Concat QDQ bucket with:

```text
selected_candidate: B3_split1_concat_lut_scalar_add
mean_total_us: 125229
mean_merge_us: 20953.9
conv_share_pct: 49.5835
activation_share_pct: 26.0505
merge_share_pct: 16.7325
output_quantize_share_pct: 5.54777
mismatches: 0
```

Stage25 must not expand the graph. It must replay Stage24 and decide one local Conv-focused lane based on measured Conv sub-buckets.

## Hard Boundaries

Do not implement full YOLO26 inference, a graph-wide scheduler, camera/full-image path, COCO/mAP, model FPS, production/default backend, `/data/ncnn` mutation, XSlim, `vmadot1/2/3`, `vmadotn`, FP/vfmadot, CPU4-7 IME, or OpenMP/all-core dispatch.

## Required Gates

1. Verify expected start head from Stage24 final response.
2. Replay Stage24 same-input ONNX cut path:

```text
mode: ime_threaded
output_quantize: rvv
merge_repair: split1_lut
warmup=10 runs=100 repeats=5
taskset -c 0-3
mismatches=0
```

3. Attribute Conv sub-buckets:

```text
branch0_conv_us
branch1_conv_us
model4_cv2_conv_us
correction_us
thread_overhead_us
pack/im2col if separable
```

4. Choose exactly one lane:

```text
C1: propagate or tune threaded Conv for the dominant selected Conv node if correctness/oracle is already available.
C2: single-thread MMT4D/tile tuning for the dominant selected Conv node.
C3: stop and write vmadot1/2/3 proof-lane recommendation if MMT4D/threading evidence shows structural low-K bottleneck, but do not implement vmadot1/2/3 unless a separate authorized stage says so.
```

5. Preserve same-input ONNX cut equality:

```text
mismatches=0
max_abs_diff=0
SHA stable
ambient frm sweep pass
```

## Reports

Create:

```text
STAGE25_FINAL_REPORT.md
STAGE25_SUMMARY_RU.md
stage24_replay_report.md
conv_bucket_attribution_report.md
conv_lane_selection_report.md
selected_conv_candidate_report.md
runner_api_vs_onnx_cut_report.md
runner_rounding_mode_report.md
source_hygiene_report.md
stage26_prompt.md
```
