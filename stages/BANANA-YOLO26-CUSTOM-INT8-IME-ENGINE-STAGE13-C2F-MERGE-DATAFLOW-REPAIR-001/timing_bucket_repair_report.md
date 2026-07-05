# Timing Bucket Repair Report

## Old Mapping

Stage 12 `finalize_timing()` added `split_us` into `pack_layout_us`, so
`pack_layout_share_pct` was a cumulative overlap rather than a non-overlapping
component share.

## New Mapping

Stage 13 adds explicit fields:

- `split_copy_us`
- `add_compute_us`
- `concat_materialize_us`
- `pack_for_model2_cv2_us`
- `layout_copy_us`
- `merge_total_us`
- `merge_share_pct`

The compatibility fields `add_us` and `concat_us` remain aliases for
`add_compute_us` and `concat_materialize_us`. `split_us` remains cumulative
for legacy output, but `split_copy_us` is the Stage 13 non-overlapping merge
bucket used in reports.

## Result

Stage 12 reported `pack_layout_share_pct=22.3855`. With repaired buckets the
same family of path reports approximately `0.14-0.16%` pack/layout share. The
real Stage 13 target is therefore merge materialization and post-Concat QDQ,
not Conv pack-layout.
