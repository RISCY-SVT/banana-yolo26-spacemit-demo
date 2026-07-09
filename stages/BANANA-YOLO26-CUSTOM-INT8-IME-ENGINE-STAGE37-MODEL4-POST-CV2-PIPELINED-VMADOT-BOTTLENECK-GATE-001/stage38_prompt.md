# BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE38-POST-3X3-PIPELINED-BOTTLENECK-GATE-001

## Mission

Continue only in `/data/banana-yolo26-spacemit-demo` on branch `yolo26-custom-int8-engine`.

Stage37 selected:

```text
mode: Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
mean_total_us: 32307.4
selected_cut_total_speedup_vs_stage36_replay: 1.107313x
combined_branch3x3_compute_speedup: 1.433051x
```

Stage38 must replay Stage37 selected mode, rebuild the non-overlapping bucket map, and select exactly one next local lane. Do not pre-select a lane without replay.

## Hard Boundaries

Do not implement full YOLO26 inference, graph-wide scheduler, graph expansion, full-image/camera path, COCO/mAP, model FPS claim, production/default-backend claim, `/data/ncnn` mutation, XSlim, `smt.vmadotus` selected path, `vmadot1/2/3` direct/sliding integration, `vmadotn`, `vfmadot`, int4/int16 paths, CPU4-7 IME execution, `rdcycle` timing, or OpenMP/all-core default dispatch.

## Replay Gate

Replay:

```text
mode: Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
warmup: 10
runs: 100
repeats: 5
affinity: taskset -c 0-3
expected_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
FRM sweep: RNE RTZ RDN RUP RMM
```

Required:

```text
mismatches=0
max_abs_diff=0
SHA stable
FRM sweep pass
attribution_pct >= 99%
```

## Lane Rules

Select exactly one:

```text
Lane A: output QuantizeLinear repair
  Select if output_quantize_us remains >=18-20% of total and a local exact RNE/RVV candidate exists.

Lane B: remaining Conv/thread-overhead repair
  Select if Conv remains dominant and a bounded >=5% selected-cut total lane exists.

Lane C: timing instrumentation/im2col attribution repair
  Select if im2col/pack remains ambiguous and blocks defensible decisions.

Lane D: no local repair; next boundary planning
  Select if no local lane has credible >=5% selected-cut total improvement.
```

## Non-Claims

Stage38 remains selected `/model.4` ONNX-cut work only. It is not full YOLO26 inference, not model FPS, not full-image/camera performance, not COCO/mAP, and not production readiness.
