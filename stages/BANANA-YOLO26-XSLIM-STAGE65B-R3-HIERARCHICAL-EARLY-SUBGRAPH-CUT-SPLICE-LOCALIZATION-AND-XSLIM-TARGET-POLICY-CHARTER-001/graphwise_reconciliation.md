# Graphwise reconciliation

R1 Graphwise and the independent H500 activation audit agree on the two task-causal regions. Exact source tensor names, not shape similarity, were used.

| Tensor | SNR | MSE | Cosine | Graphwise FP range | Graphwise Q range |
|---|---:|---:|---:|---:|---:|
| `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_output_0` | 0.1697 | 0.5738 | 0.9113 | -51.390..93.850 | -9.735..83.234 |
| `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul_output_0` | 0.1437 | 0.9129 | 0.9356 | -0.278..71.352 | -0.369..59.825 |
| `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul_output_0` | 0.1184 | 0.6864 | 0.9425 | -0.278..50.795 | 0.000..57.589 |
| `/model.2/cv2/act/Mul_output_0` | 0.0267 | 0.0615 | 0.9866 | -0.278..19.521 | -0.277..17.388 |

The H500 audit additionally measures clipping/rail fractions over 500 calibration-disjoint images. It finds normalized MAE 0.1679 at model.2, 0.2912 at P4 confidence's last prefix activation, and 0.3871 at P5 confidence's last prefix activation. The task-level cut/splice recovery, not these correlation metrics, is the causal authority.
