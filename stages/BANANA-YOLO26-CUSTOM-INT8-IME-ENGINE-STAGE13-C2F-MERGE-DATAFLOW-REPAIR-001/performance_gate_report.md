# Performance Gate Report

## Minimum Gate

| gate | result |
|---|---|
| correctness `mismatches=0` | pass |
| `total_us < 582039` | pass, `502570` |
| `pack_layout_share_pct < 15` or repaired bucket explanation | pass, `0.162724`; Stage 12 was overlapping |
| `merge_total_us < Stage12/A0 merge_total` | pass, `217677 -> 139874` |
| no activation regression > 40% | pass, `17.6073%` |
| board CPU0/1/2/3 pass | pass |

## Good Gate

| gate | result |
|---|---|
| `total_us < 450000` | not met |
| `pack_layout_share_pct < 10` | pass |
| `merge_total_us reduced by >=25%` | pass, approximately `35.7%` vs A0 |

## Decision

Stage 13 meets the minimum gate and the merge-reduction portion of the good
gate. It is ready for the next bounded block expansion, with the caveat that
future block integration should keep merge/QDQ buckets first-class.
