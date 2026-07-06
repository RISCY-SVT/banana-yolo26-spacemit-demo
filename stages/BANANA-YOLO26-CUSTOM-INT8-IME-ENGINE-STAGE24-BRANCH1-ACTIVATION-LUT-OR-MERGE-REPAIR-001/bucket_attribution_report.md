# bucket_attribution_report

## Bucket Definition

Non-overlapping accepted buckets for Stage24 replay:

```text
input_adapter_us:
  ONNX-cut uint8 input split/adaptation into local runner split buffers.

conv_us:
  Branch0 threaded Conv, branch1 Conv, model4 cv2 Conv, and their correction bucket as recorded by the runner.

activation_requant_us:
  Existing branch activation/requant work outside merge/post-Concat QDQ.

merge_us:
  Add/Concat/post-Concat QDQ selected runner region.
  This includes add_us, concat_us, and post_qdq_us as a single cumulative runner bucket.
  It is counted once in attribution.

output_quantize_us:
  Final /model.4/cv2 corrected int32 -> uint8 NHWC QuantizeLinear boundary.

copy_layout_us and pack_layout_us:
  Reported by the runner when present; both are zero in this cut path replay.

other_us:
  Directly measured total_us minus named non-overlapping buckets.
```

## Baseline Replay

```text
mean_total_us: 147624
mean_attributed_us: 147609
mean_attribution_pct: 99.9898
mean_other_us: 15.0449
```

## Stage24 Candidate

```text
mean_total_us: 125229
mean_attributed_us: 125213
mean_attribution_pct: 99.9874
mean_other_us: 15.7977
```

Attribution gate `>=99.0%`: `pass`.

No double counting was used between `merge_us`, `output_quantize_us`, and layout/copy buckets.
