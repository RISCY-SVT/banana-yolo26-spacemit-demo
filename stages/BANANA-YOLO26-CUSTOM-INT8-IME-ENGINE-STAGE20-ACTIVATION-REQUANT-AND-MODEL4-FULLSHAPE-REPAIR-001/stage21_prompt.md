# BANANA-YOLO26 Custom INT8 IME Engine Stage 21 Prompt

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

## Mission

Integrate the Stage20 `C2_split0_concat_lut_4t` representative/full-shape model4 C2f merge repair into the narrow model4 C2f runner path, preserving exact correctness and keeping representative/full-shape timing as the gate.

## Required starting evidence

Stage20 selected:

```text
selected_repair_lane: C2
selected_candidate: C2_split0_concat_lut_4t
B1_threaded_branch0_4t mean_total_us: 149539
C2_split0_concat_lut_4t mean_total_us: 116338
B1_threaded_branch0_4t mean_merge_us: 66564.3
C2_split0_concat_lut_4t mean_merge_us: 29791.6
mismatches: 0
```

## Boundaries

Do not implement full YOLO26 inference, graph-wide scheduler, default backend, camera/full-image path, COCO/mAP, XSlim, vmadot1/2/3, vmadotn, FP/vfmadot, or CPU4-7 IME.

## Next gate

After integrating the C2 repair into the narrow runner, rerun:

```text
full-shape model4 C2f correctness
warmup=10 runs=100 repeats=5
board CPU0-3 only
```

Then decide between next graph expansion and further dataflow/memory-planner work.
