# Bidirectional localization decision

Forward FQ8 and reverse QF controls passed exact reconstruction on all H500 images.
QF-C0 remains 0.007204 mAP below H8 (95% CI 0.001506..0.012273), so material error is already present by the first complete coarse frontier. QF-C6 is statistically indistinguishable from QF-C0, while QF-L3 loses another 0.008579 mAP (95% CI 0.003159..0.010965). This confirms two distributed contributors: the input/stem-to-model.2 region and the model.23 per-scale head prefixes. No single refined step reaches the predeclared quarter-residual threshold.
