# Full Output Comparison Report

Stage40 produced a final `output0` through two skeleton routes:

1. All-ORT fallback skeleton.
2. Custom `/model.4` C++ runner insertion plus ORT CPU suffix cut.

Both routes matched the full ORT CPU reference exactly.

| comparison | status | mismatches | max_abs_diff |
|---|---:|---:|---:|
| `all_ort_final_vs_full_reference` | pass | 0 | 0 |
| `custom_model4_skeleton_final_vs_full_reference` | pass | 0 | 0 |

Final output:

```text
name: output0
shape: 1x300x6
dtype: float32
array_sha256: 8ddc0e17ab7307ac7fc1f91d9145acf3f88647d7528e73183b8e6d723c41ebac
npy_sha256: d07f34ed645101cf735dd82ea10b9488f8abdc847d431d108ec78154eb238fe7
```

This is a same-input skeleton correctness result, not full-image/camera output validation and not a production claim.
