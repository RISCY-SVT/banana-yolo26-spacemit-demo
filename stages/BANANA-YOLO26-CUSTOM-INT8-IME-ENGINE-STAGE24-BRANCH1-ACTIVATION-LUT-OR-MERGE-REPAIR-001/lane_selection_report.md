# lane_selection_report

## Replay Buckets

```text
conv_share_pct: 42.0763
activation_share_pct: 22.0935
merge_share_pct: 29.3546
output_quantize_share_pct: 4.69731
```

## Lane Decision

selected_lane: `B`
selected_candidate: `B3_split1_concat_lut_scalar_add`

Stage23 replay drifted slightly from the Stage23 final report:

```text
Stage23 final merge_share_pct: 31.494
Stage24 replay merge_share_pct: 29.3546
```

Lane B was still selected because:

```text
- merge/post-Concat QDQ remained the largest safely local non-Conv repair bucket;
- activation_requant_share was below 30%;
- conv_share was 42.0763%, below the Stage24 Conv stop threshold of 45-50%;
- graph expansion is forbidden in Stage24.
```

## Rejected Lanes

Lane A was rejected because activation/requant was `22.0935%`, below the `>=30%` gate and below merge as a local repair target.

Lane C was rejected for implementation in Stage24 because Conv was below the pre-registered `45-50%` stop threshold during replay. After B3, Conv became the next decision bucket and should be handled in Stage25.

Lane D graph expansion was rejected by Stage24 hard scope.
