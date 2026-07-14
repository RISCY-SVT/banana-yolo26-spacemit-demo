# Prepared static schedule

S4 precomputes active-worker masks and operation/range dependencies, then advances a persistent worker sequence with epoch barriers. It is exact but did not improve the complete model, so raw epoch-spin remains the low-latency research route.
