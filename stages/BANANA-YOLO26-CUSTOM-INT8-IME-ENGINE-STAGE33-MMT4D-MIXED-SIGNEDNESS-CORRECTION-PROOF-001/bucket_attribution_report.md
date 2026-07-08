# Bucket Attribution Report

All timings are selected `/model.4` ONNX-cut timings, not full-model timings.

## Baseline

```text
total_us: 40380.4
conv_us: 25447.5
activation_requant_us: 2980.39
merge_us: 2133.42
output_quantize_us: 7071.04
mean_attributed_us: 40355.0
mean_attribution_pct: 99.9371
other_us: 25.4132
```

## Candidate

```text
total_us: 40934.1
conv_us: 26091.3
activation_requant_us: 2996.16
merge_us: 2174.65
output_quantize_us: 7032.32
mean_attributed_us: 40907.8
mean_attribution_pct: 99.9360
other_us: 26.2144
```

The mixed signedness candidate removes the separate `/model.4/cv2` correction timing bucket, but moves work into pack/compute/copy in the fused path. Net selected-cut time regresses.
