# next_step_decision

## Post-Repair Bucket Map

```text
mean_total_us: 125229
conv_share_pct: 49.5835
activation_share_pct: 26.0505
merge_share_pct: 16.7325
output_quantize_share_pct: 5.54777
```

## Decision

next_recommended_step: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001`

Reason:

```text
- output_quantize was repaired in Stage23 and remains small;
- merge/post-Concat QDQ was repaired in Stage24 and is no longer the largest bucket;
- activation/requant is material but below the 30% Stage24 activation lane gate;
- Conv is now the largest bucket at 49.5835%.
```

Stage25 should not expand the graph by default. It should replay Stage24, attribute Conv sub-buckets, then decide one of:

```text
- propagate/adjust threaded Conv for model4 cv2 if safe;
- MMT4D/tile tuning for current selected Conv shapes;
- open a separate vmadot1/2/3 proof lane only if explicitly authorized and justified by Conv roofline evidence.
```

No model FPS, full-image/camera performance, COCO/mAP, production readiness, or default backend claim is made.
