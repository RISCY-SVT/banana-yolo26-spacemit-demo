# CV2 Pipelined Microkernel Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Target

```text
node: /model.4/cv2/conv/Conv
shape: 1x1, 80x80, C_in=96, C_out=128
current selected path: signed-storage s8xs8 MMT4D + explicit correction
```

## Stage35 Decision

Stage35 did not implement a cv2 pipelined microkernel candidate.

Reason:

```text
The task's primary blocker was SIGILL localization and emission repair.
After the diagnostic was repaired, Step-0 throughput became board-executable and showed that independent accumulator loops are promising.
Integrating a real cv2 candidate requires a separate bounded candidate stage with same-input ONNX-cut correctness and full selected-cut benchmarking.
```

## Status

```text
cv2_C0_replay: not_run_in_stage35
cv2_C1_pipeline_2_accumulator_groups: not_implemented
cv2_C2_pipeline_4_accumulator_groups: not_implemented
cv2_C3_prefetch_distance: not_implemented
same_input_ONNX_cut_for_candidate: not_run
selected_cut_total_speedup: not_measured
```

## Recommended Candidate Basis

Use Stage35 throughput evidence to open a Stage36 candidate:

```text
candidate_basis:
  raw/proven smt.vmadot emission
  4 or 6 independent accumulator groups if register pressure allows
  signed-storage s8xs8 only
  keep existing explicit correction
  compare against current Stage26/28/33 selected path in same session
```

Hard gates for the next candidate remain:

```text
same-input ONNX cut SHA unchanged
mismatches=0
max_abs_diff=0
FRM sweep pass
CPU0-3 only
no CPU4-7 IME
no full engine / no model FPS
```
