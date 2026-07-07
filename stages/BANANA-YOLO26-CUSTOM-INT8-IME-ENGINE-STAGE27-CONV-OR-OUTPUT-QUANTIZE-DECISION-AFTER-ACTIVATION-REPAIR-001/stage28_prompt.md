# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001

Work in:

```text
/data/banana-yolo26-spacemit-demo
branch: yolo26-custom-int8-engine
expected_start_head: <Stage27 end_head>
```

## Mission

Perform a narrow Conv/MMT4D tile/prepack/correction repair stage for the existing same-input `/model.4` ONNX-cut selected path. Do not expand the graph and do not implement `vmadot1/2/3`.

Stage27 selected:

```text
SELECT_C4_TILE_PREPACK_FUTURE_STAGE
```

because Stage26 replay after activation repair showed:

```text
mean_total_us: 41669.2
conv_us: 26869.6
conv_share_pct: 64.4832
output_quantize_share_pct: 16.9765
mismatches: 0
max_abs_diff: 0
frm_sweep: pass
```

## Hard Boundaries

Do not:

```text
- implement full YOLO26 engine
- create graph-wide scheduler
- expand beyond selected /model.4 ONNX-cut path
- run full-image/camera/COCO/mAP
- claim model FPS or production readiness
- mutate /data/ncnn or YOLO11 production repo
- use XSlim
- implement vmadot1/2/3 or vmadotn
- use FP/vfmadot
- run IME on CPU4-7
- enable OpenMP/all-core/default dispatch
- push without explicit authorization
```

## Required Replay

Replay Stage27 selected path first:

```text
bench_stage23_model4_runner_cut
--mode ime_threaded
--output-quantize rvv
--merge-repair branch1_add_lut
--thread-branch0 4
--thread-branch1 4
--thread-model4-cv2 4
--warmup 10 --runs 100 --repeats 5 --frm-sweep
```

Require:

```text
mismatches=0
max_abs_diff=0
output SHA 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
FRM sweep pass
CPU0-3 only
```

## Candidate Scope

Test one bounded MMT4D/tile/prepack/correction lane at a time. Candidate examples:

```text
T1: separate pack/im2col/correction timing per Conv node
T2: correction fusion or reduced correction copy for threaded Conv outputs
T3: 1x1-specific MMT4D loop-order/tile candidate for /model.4/cv2/conv/Conv
T4: 3x3 K-blocking or output-channel blocking candidate for branch Conv nodes
T5: prepack/workspace layout reduction if measurements show it is material
```

Do not run a broad search. Pick one candidate after measurement.

## Required Reports

```text
STAGE28_FINAL_REPORT.md
STAGE28_SUMMARY_RU.md
stage27_replay_report.md
conv_component_split_report.md
tile_prepack_candidate_decision.md
selected_tile_candidate_report.md
onnx_cut_correctness_report.md
frm_rounding_regression_report.md
component_timing_report.md
source_hygiene_report.md
stage29_prompt.md
```

## Future Note

If Stage28 proves current MMT4D remains structurally limited after a focused tile/prepack attempt, recommend:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-VMADOT123-SEMANTICS-AND-CONV-APPLICABILITY-001
```

as a separate proof-only lane.
