# Exact Attention N-Tail Contract

Stage61 changes only the execution route for exact Q0 MatMul shapes whose K or
N dimension is not a multiple of the existing IME panel width. It does not
change graph data, qparams, arithmetic, Softmax tables, output encoding, or tie
ordering.

## Decomposition

Complete N16 blocks call the existing `y26_stage48_kernel_m12n16` symbol.
The live tail uses:

| Live columns | Route |
| ---: | --- |
| 1..4 | one N4 call |
| 5..8 | one N8 call |
| 9..12 | N8 plus N4 |
| 13..15 | N8 plus N8 (selected candidate); padded N16 retained as tested alternative |
| 16 | one N16 call |

K is padded to eight lanes only inside the packed panel. Dead K lanes and dead
N columns use the operation's signed-storage zero-point value. Correction sums
and the constant term use the padded K consistently, so the dead contributions
cancel algebraically. Only live columns are requantized and stored.

## Preserved Invariants

- `K1X_INT8_V1`, int32 accumulation, Q62 M63/RNE and saturation
- exact left/right zero-point correction
- exact Q48 Softmax and direct second-MatMul packing
- output storage bytes and deterministic TopK/tie order
- ambient FRM, `vcsr`, `vxrm` and `vxsat` restoration
- IME execution on CPU0-3 only

The independent arbitrary-precision property matrix covers N=1..31 and all
required boundary values through 577, K/M tails, asymmetric zero points, INT8
extremes, positive/negative accumulators, RNE ties, saturation, red zones,
invalid sizes, alias rejection, and both N13 strategies.
