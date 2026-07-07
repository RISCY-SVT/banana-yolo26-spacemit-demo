# Output Quantize Secondary Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE27-CONV-OR-OUTPUT-QUANTIZE-DECISION-AFTER-ACTIVATION-REPAIR-001

## Replay Evidence

Stage26 accepted path replay:

```text
output_quantize_us: 7073.94
output_quantize_share_pct: 16.9765
```

Same-session all-4 matrix replay:

```text
output_quantize_us: 7081.21
output_quantize_share_pct: 17.0047
```

## Decision

Output quantize remains material, but it is below the Stage27 `>20%` secondary-lane threshold. Conv remains dominant at about `64.5%`, so Stage27 does not implement another output-quantize repair.

If a future Conv stage substantially reduces Conv time, output quantize should be rechecked before graph expansion.
