# Activation Approximation Future Lane

classification: deferred

Stage 9 did not implement:

- PWL SiLU approximation.
- hard-swish.
- ReLU replacement.
- activation NAS.
- QAT/retraining.
- model rewrite.
- DULUT or large accumulator-domain LUT.

Reason:

Exact A2 RVV f32 requant + 256-entry LUT already passed correctness and reduced activation share below the Stage 9 gate. Approximation work is not needed for this selected-subset gate and would require a separate tolerance and accuracy policy.
