# Next Step Decision

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

## Post-Repair Buckets

```text
mean_total_us: 137547
conv_share_pct: 37.9583
activation_share_pct: 23.7138
merge_share_pct: 31.494
output_quantize_share_pct: 4.97977
mismatches: 0
```

## Decision

```text
next_recommended_step: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE24-BRANCH1-ACTIVATION-LUT-OR-MERGE-REPAIR-001
```

Reason:

```text
The final output quantization blocker is repaired.
Conv is the largest remaining bucket at 37.96%, but below a clear Conv-only gate.
Merge remains material at 31.49% and should be inspected before graph expansion.
Activation is also material at 23.71%, so Stage24 should compare a branch1 activation LUT/RVV repair against a merge-focused repair and select exactly one measured lane.
```

Do not expand the graph before Stage24 rechecks the post-repair bucket map through the real runner API.
