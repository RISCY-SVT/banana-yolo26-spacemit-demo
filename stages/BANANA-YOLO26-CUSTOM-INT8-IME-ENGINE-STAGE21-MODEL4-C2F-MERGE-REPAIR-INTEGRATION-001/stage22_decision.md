# Stage22 Decision

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

## Measured State

```text
representative/full-shape selected subset total_us: 116631
conv_share_pct: 45.5842
activation_share_pct: 25.5398
merge_share_pct: 25.5224
mismatches: 0
```

## Decision Rules Applied

```text
Conv >45-50%: borderline yes
Activation/requant >30%: no
Merge/dataflow >30%: no
Buckets balanced and correctness stable: mostly yes
Direct full-shape ONNX cut for integrated runner: still needed
```

## Recommended Stage22

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001
```

Stage22 should first close the direct full-shape ONNX oracle cut for the integrated runner, then choose between:

```text
1. next graph expansion if oracle/correctness remains stable;
2. /model.4/cv2 Conv/IME tuning if Conv share rises above 50%;
3. activation or merge repair only if those shares regress above 30%.
```

Compact-only performance work should not be used for model4 full-shape decisions.
