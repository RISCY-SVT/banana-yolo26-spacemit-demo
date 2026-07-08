# Next Stage Decision

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Decision

```text
decision: OPEN_STAGE36_BOUNDED_CV2_PIPELINED_VMADOT_CANDIDATE
```

## Basis

Stage35 repaired the invalid Stage34 diagnostic path:

```text
root_cause: rdcycle measurement SIGILL
exact helper-shaped named case: board-executable
exact helper-shaped raw case: board-executable
standalone named/raw: board-executable
2/4/6 independent accumulator raw groups: board-executable
CPU1/2/3 key smoke: pass
```

Throughput diagnostic:

```text
independent accumulator groups improve microbench throughput
register shape ceiling not observed for tested shapes
cv2 real-node candidate not implemented in Stage35
```

## Recommended Stage36

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE36-CV2-PIPELINED-VMADOT-CANDIDATE-001
```

Stage36 scope:

```text
- target only /model.4/cv2/conv/Conv
- keep signed-storage s8xs8 smt.vmadot
- keep explicit correction
- implement one bounded 4- or 6-accumulator pipelined candidate
- no smt.vmadotus selected path
- no vmadot1/2/3 direct/sliding lane
- no vmadotn / vfmadot
- no graph expansion
- compare against current selected path in same board session
```

Required Stage36 gates:

```text
same-input ONNX-cut mismatches=0
output SHA unchanged
FRM sweep pass
CPU0-3 only
host CTest pass
RISC-V cross build pass
board stable benchmark pass
minimum cv2 compute speedup >=1.25x
minimum selected-cut total speedup >=1.05x
```

## Non-Claims

This decision is not full YOLO26 inference, not model FPS, not camera/full-image performance, not COCO/mAP, not production readiness, and not default backend readiness.
