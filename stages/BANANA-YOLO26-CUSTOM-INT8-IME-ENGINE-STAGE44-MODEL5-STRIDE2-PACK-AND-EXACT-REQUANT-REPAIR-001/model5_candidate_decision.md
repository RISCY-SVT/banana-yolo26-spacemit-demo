# Model5 candidate decision

Selected experimental candidate: `R2a_stride2_chunk_fastpack`.

R2a adds an explicit non-default dataflow mode. For exact 3x3 stride-2, pad-1, channels-divisible-by-8 tiles that do not cross an output row, it copies eight channel bytes at a time into the existing MMT4D panel and uses a bounded zero-point border path. R0 remains available and unchanged.

R2a is exact on F0-F7 and workers 1-4 for F0. The paired local model5 test improves `1.94291%`, but R2a is still `106.4537%` slower than resource-matched ORT intra4. The final hybrid scaffold is `0.821898%` slower than model4-only. R2a is retained as experimental evidence, not selected acceleration.

R1, full LUT-fused R2, R3, and R4 were not stacked. Even eliminating the measured pack bucket entirely would not close the resource-matched gap, and Stage44 authorizes one bounded repair rather than an open-ended search.
