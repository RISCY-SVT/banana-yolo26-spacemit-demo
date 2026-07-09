# Per-Bucket Attribution Report

## Attribution Policy

Buckets are non-overlapping for selected-cut total attribution:

- `input_adapter_us`, `conv_us`, `activation_requant_us`, `merge_us`, and `output_quantize_us` are added once.
- `thread_overhead_us` is diagnostic and already included inside `conv_us`; it is not added again.
- `conv_compute_us` is the bench raw worker compute field. For branch 3x3 analysis, `im2col_pack_us` is split out as a sub-bucket and excluded from derived `mmt4d_compute_excluding_pack_us`.

## Stage37 Replay

| bucket | mean_us | share_pct |
|---|---:|---:|
| input_adapter_us | 2640.44 | 8.028 |
| conv_us | 18051.0 | 54.8823 |
| activation_requant_us | 3004.23 | 9.13404 |
| merge_us | 2112.38 | 6.42248 |
| output_quantize_us | 7055.2 | 21.4506 |
| other_us | 27.1973 | 0.0827 |
| total_us | 32890.5 | 100.0 |
| attribution_pct | 99.9173 | |

## Stage38 Lane A Candidate

| bucket | mean_us | share_pct |
|---|---:|---:|
| input_adapter_us | 2602.06 | 8.576 |
| conv_us | 18048.1 | 59.4832 |
| activation_requant_us | 3036.11 | 10.0064 |
| merge_us | 2075.46 | 6.84031 |
| output_quantize_us | 4551.97 | 15.0024 |
| other_us | 27.8229 | 0.0917 |
| total_us | 30341.5 | 100.0 |
| attribution_pct | 99.9083 | |

## Result

- bucket_attribution_status: `pass`
- replay attribution is above the required `98%` threshold.
- selected candidate attribution remains above the required `98%` threshold.
