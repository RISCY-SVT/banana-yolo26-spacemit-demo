# Stage21 Short Roofline Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE21-MODEL4-C2F-MERGE-REPAIR-INTEGRATION-001`

This is selected-subset evidence only. It is not full YOLO26 utilization and not model FPS.

## Current Representative/Full-Shape C2 Buckets

```text
total_us: 116631
conv_us: 53165.3
activation_requant_us: 29787.3
merge_us: 29767
conv_share_pct: 45.5842
activation_share_pct: 25.5398
merge_share_pct: 25.5224
```

## Conv Notes

The largest Conv bucket in the current selected path remains `/model.4/cv2/conv/Conv`:

```text
model4_cv2_conv_us: 31116.7
branch1_conv_us: 15884.2
branch0_conv_us: 6164.37
```

The selected subset is now relatively balanced between Conv, activation/requant, and merge/dataflow. No single non-Conv bucket exceeds 30%.

## Likely Next Bottleneck

```text
primary: Conv / IME, especially /model.4/cv2/conv/Conv
secondary: activation/requant and merge/dataflow are both material but below 30%
```

`vmadot1/2/3` remains a future separate proof lane only. Stage21 did not implement sliding variants.
