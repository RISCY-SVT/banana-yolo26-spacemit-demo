# K1X INT8 Executor Correctness

The deployment authority is `K1X_INT8_V1`, not legacy float-QDQ ONNX Runtime.
The hierarchy is:

1. Independent Python arbitrary-precision operator oracles.
2. Portable C++ scalar arithmetic and graph execution.
3. Board scalar execution.
4. Board optimized K1X execution.

Required equality is byte-exact at integer boundaries. Final `1x300x6` output
is exact within the frozen profile's fixed-point box/score representation and
tie policy.

Tests cover positive and negative rounding ties, threshold neighborhoods,
saturation, accumulator bounds, padding, spatial/channel tails, N4/N8/N16
heads, Resize edges, Softmax normalization, TopK ties, arena lifetimes, and
package corruption. Ambient FRM and vector fixed-point CSR state must be
restored after every run. CPU4-7 IME count must remain zero.

Legacy float-QDQ outputs are diagnostic only. A difference from backend float
accumulation does not by itself violate the integer contract; task accuracy is
validated independently on COCO val2017.
