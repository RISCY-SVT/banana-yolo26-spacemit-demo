# Component Timing Report

## Representative/full-shape Stage18 replay

| candidate | total_us | conv_us | activation_requant_us | split_us | correction_us | conv_share_pct | activation_share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_stage17_single_thread_replay | 25588.139540 | 20402.617866 | 5000.416242 | 183.651952 | 216.086072 | 79.734667 | 19.541930 |
| A4_integrated_threaded_conv_4t | 11082.483550 | 5905.210462 | 4983.945734 | 189.137874 | 174.622230 | 53.284180 | 44.971379 |

## Compact Stage19 C2f oracle scope

| candidate | total_us | conv_us | activation_requant_us | split_us | add_us | concat_us | post_concat_qdq_us | pack_layout_us | correction_us | thread_overhead_us | conv_share_pct | activation_share_pct | merge_share_pct | pack_layout_share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_single_thread_c2f | 186.079392 | 122.624718 | 27.298578 | 9.192498 | 10.972322 | 10.972322 | 13.745448 | 0.275334 | 2.588628 | 0.000000 | 65.899139 | 14.670393 | 12.326967 | 0.147966 |
| A4_threaded_conv_4t | 283.534780 | 212.156010 | 32.392534 | 9.311166 | 10.850838 | 10.850838 | 13.657124 | 0.352684 | 3.915530 | 76.068746 | 74.825392 | 11.424536 | 8.100696 | 0.124388 |
| A5_threaded_conv_threaded_activation_4t | 461.796518 | 204.429350 | 216.748804 | 9.700472 | 10.884824 | 10.884824 | 13.792054 | 0.390838 | 3.971882 | 258.251034 | 44.268274 | 46.935998 | 5.087203 | 0.084634 |

Timing caveat:

```text
Stage19 C2f timing is compact oracle-scope evidence only.
It must not be compared as full-shape timing or full-model timing.
```
