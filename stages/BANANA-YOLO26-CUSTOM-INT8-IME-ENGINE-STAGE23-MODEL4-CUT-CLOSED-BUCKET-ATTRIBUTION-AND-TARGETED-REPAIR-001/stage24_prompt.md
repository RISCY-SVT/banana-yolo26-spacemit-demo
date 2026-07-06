# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE24-BRANCH1-ACTIVATION-LUT-OR-MERGE-REPAIR-001

You are Codex working in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine`.

User-facing summaries must be in Russian. Code, comments, commands, paths, identifiers, filenames, and artifact names stay in English.

## Mission

Stage23 closed the real runner API same-input ONNX cut path for `/model.4` and repaired final `/model.4/cv2` output quantization with exact RVV output QuantizeLinear.

Stage24 must use the Stage23 post-repair bucket map to choose exactly one next local repair lane before graph expansion.

Stage23 post-repair evidence:

```text
mean_total_us: 137547
stddev_total_us: 81.7884
mismatches: 0
conv_share_pct: 37.9583
activation_share_pct: 23.7138
merge_share_pct: 31.494
output_quantize_share_pct: 4.97977
```

## Hard boundaries

Do not implement full YOLO26 inference, a graph-wide scheduler, a default backend switch, camera/full-image path, COCO/mAP, model FPS, production claims, `/data/ncnn` mutation, XSlim, vmadot1/2/3, vmadotn, FP/vfmadot, CPU4-7 IME, or OpenMP/all-core default dispatch.

## Required gates

1. Reproduce Stage23 real runner API ONNX-cut correctness:

```text
y26_stage16_model4_c2f_run_cut_u8_output
input: /model.4/cv1/conv/Conv_output_0_QuantizeLinear_Output
output: /model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output
mismatches=0
```

2. Reproduce Stage23 stable timing protocol:

```text
taskset -c 0-3
warmup=10
runs=100
repeats=5
```

3. Choose exactly one repair lane:

```text
Lane A: branch1 activation/requant LUT or RVV repair if activation bucket is the best local target.
Lane B: merge/post-Concat QDQ/dataflow repair if merge bucket remains >30%.
Lane C: Conv/threading propagation only if Conv rises above a clear >45-50% gate.
Lane D: stop and recommend graph expansion only if buckets are balanced and no local repair is justified.
```

4. Preserve ONNX-cut byte equality:

```text
mismatches=0
max_abs_diff=0
SHA/checksum stable
```

## Required reports

```text
STAGE24_FINAL_REPORT.md
STAGE24_SUMMARY_RU.md
stage23_replay_report.md
repair_lane_decision.md
selected_repair_correctness_report.md
selected_repair_benchmark_report.md
stage25_prompt.md
source_hygiene_report.md
```

Do not expand the graph in Stage24 unless the decision report proves all local repair lanes are not material.
