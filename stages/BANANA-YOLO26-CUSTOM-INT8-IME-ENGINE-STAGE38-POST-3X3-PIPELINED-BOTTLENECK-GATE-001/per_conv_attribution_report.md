# Per-Conv Attribution Report

## Stage37 Replay

| node | conv_total_us | im2col_pack_us | raw_compute_us | mmt4d_compute_excluding_pack_us | correction_us | writeback_us |
|---|---:|---:|---:|---:|---:|---:|
| `/model.4/m.0/cv1/conv/Conv` | 5937.63 | 3646.44 | 4148.01 | 501.57 | 275.682 | 0 |
| `/model.4/m.0/cv2/conv/Conv` | 4336.77 | 2014.70 | 2827.56 | 812.86 | 442.092 | 0 |
| `/model.4/cv2/conv/Conv` | 7776.64 | 1137.78 | 3976.30 | 2838.52 | 1762.99 | 0 |

Combined branch 3x3:

- conv_total_us: `10274.4`
- im2col_pack_us: `5661.14`
- im2col share of branch conv: `55.10%`

## Stage38 Lane A Candidate

| node | conv_total_us | im2col_pack_us | raw_compute_us | mmt4d_compute_excluding_pack_us | correction_us | writeback_us |
|---|---:|---:|---:|---:|---:|---:|
| `/model.4/m.0/cv1/conv/Conv` | 5928.18 | 3601.25 | 4108.87 | 507.62 | 240.071 | 0 |
| `/model.4/m.0/cv2/conv/Conv` | 4319.00 | 2000.66 | 2809.33 | 808.67 | 441.830 | 0 |
| `/model.4/cv2/conv/Conv` | 7800.94 | 1130.18 | 3976.05 | 2845.87 | 1805.15 | 0 |

Combined branch 3x3 after Lane A:

- conv_total_us: `10247.18`
- im2col_pack_us: `5601.91`
- im2col share of branch conv: `54.67%`

## Interpretation

Stage38 selected Lane A because output QuantizeLinear met the decision-tree threshold and had a low-risk exact repair. After Lane A, branch 3x3 im2col/pack is the clearest next local Conv sub-bucket.
