# Im2col/Pack Split Report

## Instrumentation

Stage38 added optional `--measure-im2col-pack` timing to the selected-cut bench. The timing is disabled unless explicitly requested. It accumulates the pack/im2col preparation time inside the existing MMT4D worker path and reports per-node fields.

## Replay Split

| metric | value_us |
|---|---:|
| total_conv_im2col_pack_us | 6798.92 |
| branch0_im2col_pack_us | 3646.44 |
| branch1_im2col_pack_us | 2014.70 |
| model4_cv2_im2col_pack_us | 1137.78 |
| combined_branch3x3_im2col_pack_us | 5661.14 |

## Candidate Split

| metric | value_us |
|---|---:|
| total_conv_im2col_pack_us | 6732.10 |
| branch0_im2col_pack_us | 3601.25 |
| branch1_im2col_pack_us | 2000.66 |
| model4_cv2_im2col_pack_us | 1130.18 |
| combined_branch3x3_im2col_pack_us | 5601.91 |

## Gate

- im2col_split_status: `pass`
- im2col/pack is no longer hidden inside branch 3x3 compute for Stage38 reports.
- combined branch 3x3 im2col/pack remains material after Lane A: `5601.91 us`, `54.67%` of combined branch conv total.
