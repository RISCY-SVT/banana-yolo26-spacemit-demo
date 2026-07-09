# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Mission

Continue only in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine`.

Stage36 selected `Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4` for the current `/model.4` same-input ONNX-cut path. Stage37 must replay Stage36 A1, rebuild non-overlapping buckets, and select exactly one next local lane. Do not expand the graph and do not implement a full YOLO26 engine.

## Required starting facts

Recover Stage36:

```text
selected_candidate: A1_branch1_add_lut_cv2_pipelined4
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
baseline_total_us: 37341.1
stage36_a1_total_us: 33192.7
stage36_a1_model4_cv2_compute_us: 3616.14
stage36_a1_model4_cv2_compute_speedup: 2.085580x
stage36_a1_total_speedup: 1.124979x
```

## Hard boundaries

Do not:

```text
- implement full YOLO26 engine;
- expand beyond current selected /model.4 ONNX cut;
- run full-image/camera/COCO/mAP;
- claim model FPS or production/default-backend readiness;
- mutate /data/ncnn;
- use XSlim;
- use CPU4-7 for IME;
- use vmadotn;
- use smt.vmadotus as selected path without a new explicit proof;
- integrate smt.vmadot1/2/3 direct/sliding path;
- use rdcycle timing;
- push unless explicitly authorized.
```

## Gates

Replay Stage36 A1 with:

```text
taskset -c 0-3
warmup=10
runs=100
repeats=5
FRM sweep RNE/RTZ/RDN/RUP/RMM
mismatches=0
max_abs_diff=0
output SHA unchanged
```

Then rebuild per-bucket and per-Conv attribution.

## Lane selection

Choose exactly one lane:

```text
A. branch 3x3 Conv/thread-overhead repair
B. output QuantizeLinear repair
C. thread overhead/persistent-region repair
D. no local repair, prepare next graph/block or full-runner skeleton gate
```

Do not run multiple lanes in one stage.

## Required reports

Create:

```text
STAGE37_FINAL_REPORT.md
STAGE37_SUMMARY_RU.md
stage36_replay_report.md
per_bucket_attribution_report.md
per_conv_attribution_report.md
lane_selection_report.md
selected_lane_correctness_report.md
selected_lane_benchmark_report.md
source_hygiene_report.md
stage38_prompt.md
```

## Non-claims

This remains selected-cut evidence only. It is not full YOLO26 inference, model FPS, full-image/camera performance, COCO/mAP, production readiness, or default backend readiness.
