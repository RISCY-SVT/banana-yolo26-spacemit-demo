# Per-Bucket Attribution Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Attribution Policy

Buckets are non-overlapping at the current runner instrumentation level. Thread overhead is included once as `thread_overhead_us`. Per-Conv `compute_us` currently includes the existing local GEMM worker compute plus any unseparated inner pack/layout work exposed by that worker path; a separate per-node `im2col_pack_us` counter is not available in the current runner.

## Stage36 Same-Session Baseline

```text
mode: Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4
mean_total_us: 35774.4
attributed_us: 35747.9
unattributed_us: 26.4623
attribution_pct: 99.926
```

| bucket | mean_us | share_pct |
| --- | ---: | ---: |
| input_adapter_us | 2436.16 | 6.8098 |
| conv_us | 21082.8 | 58.9327 |
| activation_requant_us | 2986.17 | 8.34723 |
| merge_us | 2131.95 | 5.95942 |
| output_quantize_us | 7110.83 | 19.8769 |
| other_us | 26.4623 | 0.07397 |

## Stage37 Candidate

```text
mode: Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4
mean_total_us: 32307.4
attributed_us: 32281.2
unattributed_us: 26.2455
attribution_pct: 99.9188
```

| bucket | mean_us | share_pct |
| --- | ---: | ---: |
| input_adapter_us | 2364.62 | 7.318 |
| conv_us | 17742.7 | 54.9182 |
| activation_requant_us | 2993.66 | 9.26617 |
| merge_us | 2099.23 | 6.49766 |
| output_quantize_us | 7081.0 | 21.9176 |
| other_us | 26.2455 | 0.08124 |

## Delta

```text
total_delta_us: 3467.0
selected_cut_total_speedup: 1.107313x
conv_delta_us: 3340.1
conv_speedup: 1.18825x
output_quantize_delta_us: 29.83
```

## Interpretation

The Stage37 candidate moved the selected-cut total by 1.107313x and reduced the Conv bucket by 3340.1 us. After the candidate, Conv remains the largest bucket at 54.9182%, while output QuantizeLinear is now a material secondary bucket at 21.9176%.

## Limitation

`per_conv_im2col_pack_us` is not separately counted in the current selected runner. It is reported as `included_in_compute` in `per_conv_attribution_report.md`; this is a known attribution limitation and should be addressed before making narrower im2col-specific claims.
