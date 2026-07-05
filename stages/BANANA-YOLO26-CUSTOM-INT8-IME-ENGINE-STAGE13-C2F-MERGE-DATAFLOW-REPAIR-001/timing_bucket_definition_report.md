# Timing Bucket Definition Report

Stage 13 uses non-overlapping timing buckets:

| bucket | definition |
|---|---|
| `conv_us` | All Conv work already included in the selected subset, including `/model.2/cv2/conv/Conv`. |
| `activation_requant_us` | Existing activation/requant buckets before the Stage 12 merge. |
| `split_copy_us` | Stage 12 local split float/QDQ materialization for the C2f merge. |
| `add_compute_us` | Standalone Add compute if materialized. Fused candidates report `0`. |
| `concat_materialize_us` | Standalone Concat materialization if materialized. Fused QDQ reports `0`. |
| `post_concat_qdq_us` | Quantization from float-domain merge output to signed int8 post-Concat storage. |
| `pack_for_model2_cv2_us` | Only explicit additional pack/layout work before `/model.2/cv2/conv/Conv`; current path reports `0` because Conv packing is inside `model2_cv2_conv_us`. |
| `layout_copy_us` | Additional layout copies outside split/concat/QDQ. |
| `correction_us` | `/model.2/cv2/conv/Conv` zero-point correction. |
| `model2_cv2_conv_us` | `/model.2/cv2/conv/Conv` raw Conv plus correction timing as exposed by the current API. |
| `merge_total_us` | `split_copy_us + add_compute_us + concat_materialize_us + post_concat_qdq_us + layout_copy_us`. |
| `total_us` | Direct wall-clock selected-subset timing. |

`pack_layout_us` now excludes `split_copy_us`. Stage 12 included split in this
bucket, which overstated the actual pack/layout share.
