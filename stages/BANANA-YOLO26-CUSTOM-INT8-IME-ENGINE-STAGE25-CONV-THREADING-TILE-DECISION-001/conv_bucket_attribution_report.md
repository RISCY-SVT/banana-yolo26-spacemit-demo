# Conv Bucket Attribution Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

## Stage24 Replay Bucket Map

```text
mean_total_us: 125176
mean_conv_us: 62070.3
mean_activation_requant_us: 32592.5
mean_merge_us: 20961.5
mean_output_quantize_us: 6945.9
mean_attribution_pct: 99.9871
conv_share_pct: 49.5863
activation_share_pct: 26.0372
merge_share_pct: 16.7456
output_quantize_share_pct: 5.54889
```

Per-Conv replay:

```text
/model.4/m.0/cv1/conv/Conv: 6529.83 us
/model.4/m.0/cv2/conv/Conv: 18189.1 us
/model.4/cv2/conv/Conv: 37351.4 us
```

## Selected C1 Bucket Map

selected_candidate: C1_thread_branch1_and_model4_cv2_4t

```text
mean_total_us: 89178.9
stddev_total_us: 268.184
mean_conv_us: 26164.1
mean_activation_requant_us: 32800.7
mean_merge_us: 20964.3
mean_output_quantize_us: 6579.45
mean_thread_overhead_us: 4330.86
mean_attribution_pct: 99.977
conv_share_pct: 29.3389
activation_share_pct: 36.7808
merge_share_pct: 23.5081
output_quantize_share_pct: 7.37782
```

Per-Conv selected:

```text
/model.4/m.0/cv1/conv/Conv: 7830.04 us
/model.4/m.0/cv2/conv/Conv: 6296.98 us
/model.4/cv2/conv/Conv: 12037.1 us
```

## Decision

Conv was the dominant replay bucket before C1. After C1 threading propagation, Conv is no longer dominant; activation/requant becomes the largest selected-cut bucket.
