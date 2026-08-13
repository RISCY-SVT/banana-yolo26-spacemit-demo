# Mechanism decision

Classification: `distributed-error`, with `outlier-dominated-range`, lower-tail clipping/rail occupancy, and reconstruction error as the best-supported mechanisms.

- Early R0/model.2: model.2 output has normalized MAE 0.1679, cosine 0.9867, 3.07% FP values below the representable range, and 12.94% low-rail occupancy.
- Late R7 confidence prefixes: P4's last prefix activation has a zero real minimum, clipping 27.06% negative FP SiLU values; P5's last prefix activation is bounded near 93.80 while FP values reach 153.17. Their normalized MAE values are 0.2912 and 0.3871.
- The six-output terminal Q/DQ pairs are excluded here by the accepted D8 bypass. Therefore these measurements describe the remaining upstream residual.
- No individual refined step met the quarter-residual task threshold, so a single-op defect is not established. Attention/MatMul sensitivity and merge-qdomain mismatch remain secondary possibilities, not the selected causal classification.
