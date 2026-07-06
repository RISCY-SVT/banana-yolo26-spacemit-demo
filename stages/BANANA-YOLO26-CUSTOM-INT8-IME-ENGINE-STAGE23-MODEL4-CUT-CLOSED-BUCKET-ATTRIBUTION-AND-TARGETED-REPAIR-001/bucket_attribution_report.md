# Bucket Attribution Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

## Non-Overlapping Bucket Policy

```text
input_adapter_us:
  Build cut Split inputs from the uint8 NHWC cut input.

conv_us:
  Sum of selected Conv IME work in the model4 cut runner, including branch0, branch1, and model4 cv2 Conv components.

activation_requant_us:
  Branch activation/requant work before merge.

merge_us:
  Split/Add/Concat/post-Concat QDQ merge dataflow. This bucket includes post_concat_qdq_us by design for the selected Stage21/Stage23 merge path.

output_quantize_us:
  Final /model.4/cv2 corrected int32 output -> uint8 NHWC QuantizeLinear output.

copy_layout_us:
  Additional copy/layout work not counted above.

pack_layout_us:
  Pack/layout work not counted above.

thread_overhead_us, correction_us:
  Reported diagnostics, not included in attribution sum to avoid overlap with conv internals.

other_us:
  measured total_us - attributed_nonoverlap_us.
```

Attribution sum:

```text
input_adapter_us + conv_us + activation_requant_us + merge_us +
output_quantize_us + copy_layout_us + pack_layout_us
```

## Stage22 Baseline Context

Stage22 stable selected-cut timing:

```text
mean_total_us: 225214
mean_conv_us: 52062.6
mean_activation_requant_us: 32388.1
mean_merge_us: 42446.9
mean_thread_overhead_us: 352.271
mean_correction_us: 2371.64
```

Stage22 did not separately attribute the final output QuantizeLinear boundary required by the ONNX cut output.

## Stage23 Scalar Output Quantization Attribution

```text
mode: ime_threaded
output_quantize: scalar
warmup: 10
runs: 100
repeats: 5
mismatches: 0
mean_total_us: 205098
stddev_total_us: 179.892
cv_total_pct: 0.0877103
mean_input_adapter_us: 2583.86
mean_conv_us: 52538.5
mean_activation_requant_us: 32631.1
mean_merge_us: 43345.3
mean_output_quantize_us: 73983.9
mean_copy_layout_us: 0
mean_pack_layout_us: 0
mean_attributed_us: 205083
mean_attribution_pct: 99.9928
mean_other_us: 14.8601
conv_share_pct: 25.6164
activation_share_pct: 15.9101
merge_share_pct: 21.134
output_quantize_share_pct: 36.0726
```

## Stage23 RVV Output Quantization Attribution

```text
mode: ime_threaded
output_quantize: rvv
warmup: 10
runs: 100
repeats: 5
mismatches: 0
mean_total_us: 137547
stddev_total_us: 81.7884
cv_total_pct: 0.0594623
mean_input_adapter_us: 2535.46
mean_conv_us: 52210.3
mean_activation_requant_us: 32617.5
mean_merge_us: 43318.9
mean_output_quantize_us: 6849.5
mean_copy_layout_us: 0
mean_pack_layout_us: 0
mean_attributed_us: 137532
mean_attribution_pct: 99.9892
mean_other_us: 14.8271
conv_share_pct: 37.9583
activation_share_pct: 23.7138
merge_share_pct: 31.494
output_quantize_share_pct: 4.97977
```

## Conclusion

`H2_bucket_attribution` passed. Stage23 attributed more than 99.98% of selected-cut runtime into named non-overlapping buckets. The previously large Stage22 unattributed work was confirmed to be dominated by the final `/model.4/cv2` output quantization boundary.
