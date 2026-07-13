# Reference architecture

Correctness authority is layered:

1. Independent Python arbitrary-precision package/operator audit.
2. Portable C++ scalar full graph.
3. Board C++ scalar full graph.
4. Board optimized IME/RVV full graph.

The Python audit checks every Conv channel's Q62/M63 assets, corrected bias,
weight sum, accumulator bound, LUT dimensions, MatMul descriptors, Softmax
tables, Resize policy, and head tie contract. The complete portable C++ scalar
executor emits every integer boundary for byte comparison. ORT is not used as
the integer oracle.
