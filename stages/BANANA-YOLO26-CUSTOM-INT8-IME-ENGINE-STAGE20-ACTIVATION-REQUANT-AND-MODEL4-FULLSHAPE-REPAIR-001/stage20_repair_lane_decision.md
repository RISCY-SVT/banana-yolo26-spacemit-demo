# Stage20 Repair Lane Decision

## Bucket Evidence

Before repair, the best representative/full-shape model4 C2f threaded path was:

```text
B1_threaded_branch0_4t:
  mean_total_us: 149539
  mean_conv_us: 52979.2
  mean_activation_requant_us: 29780.3
  mean_merge_us: 66564.3
  conv_share_pct: 35.4283
  activation_share_pct: 19.9147
  merge_share_pct: 44.513
```

The merge/post-Concat-QDQ bucket exceeded the Stage20 25-35% lane threshold.

## Selected Lane

selected_repair_lane: `C2`
selected_candidate: `C2_split0_concat_lut_4t`

Reason:

```text
merge_share_pct before repair: 44.513
activation_share_pct before repair: 19.9147
conv_share_pct before repair: 35.4283
```

Lane C1 was rejected because activation/requant was not the dominant full-shape bucket. Lane C3 was rejected because Conv was not dominant after threaded branch0. Lane C4 was rejected because a local exact C2 repair was available.

## Result

```text
C2_split0_concat_lut_4t:
  mean_total_us: 116338
  mean_merge_us: 29791.6
  merge_share_pct: 25.6078
  mismatches: 0
```

Further compact-only work is not sufficient for this issue. Future work should keep representative/full-shape timing as the decision gate.
