# Activation Approximation Sidecar Report

Candidate A6 PWL SiLU: not run.

Candidate A7 hard-swish approximation: not run.

Reason: A2 `int8_lut` passed correctness and substantially reduced activation/requant time. Approximation sidecars are not needed for Stage 8 acceptance and would introduce accuracy/tolerance questions without mAP authorization.

No retraining, QAT, global model activation replacement, or mAP claim was made.
