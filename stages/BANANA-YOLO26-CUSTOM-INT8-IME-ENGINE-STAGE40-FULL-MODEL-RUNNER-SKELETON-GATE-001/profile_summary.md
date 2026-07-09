# Profile Summary

Profile source: `per_block_profile.tsv`.

This is skeleton profiling only. It is not model FPS and not production latency.

| block | implementation | mean_us | note |
|---|---|---:|---|
| full_ort_reference | ORT CPU | 198259.272 | full ORT reference |
| prefix_images_to_model4_input | ORT CPU | 56796.029 | `images -> /model.4 input` |
| model4_cut_all_ort | ORT CPU | 9466.208 | ORT fallback for closed custom block |
| suffix_model4_output_to_output0 | ORT CPU | 129572.132 | largest fallback region |

Board selected `/model.4` custom runner timing for the same deterministic boundary input:

```text
mean_total_us: 26428.9
stddev_total_us: 129.331
cv_total_pct: 0.489355
```

The board custom block timing is useful boundary evidence, but it is not full-model timing because prefix/suffix are still ORT CPU fallback in the skeleton.
