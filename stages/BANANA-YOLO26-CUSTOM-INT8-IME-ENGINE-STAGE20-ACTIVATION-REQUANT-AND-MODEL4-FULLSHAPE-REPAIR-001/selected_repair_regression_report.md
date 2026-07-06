# Selected Repair Regression Report

source_table: `selected_repair_regression.tsv`

## Summary

The C2 candidate improved total time and merge time without correctness regression:

```text
B1_threaded_branch0_4t mean_total_us: 149539
C2_split0_concat_lut_4t mean_total_us: 116338
B1_threaded_branch0_4t mean_merge_us: 66564.3
C2_split0_concat_lut_4t mean_merge_us: 29791.6
mismatches: 0
```

Conv time and activation/requant time remained essentially unchanged; the repair specifically moved the merge/post-Concat-QDQ bottleneck.
