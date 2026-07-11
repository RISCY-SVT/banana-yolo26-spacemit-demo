# Board ORT full-model forensics

The untruncated profile assigns all observed node events to CPUExecutionProvider.
It exposes operator-level execution, not inner-kernel identity. Major profiled
families include QLinearConv, DequantizeLinear, Mul, Sigmoid, QuantizeLinear,
Transpose/Concat, MaxPool, QLinearAdd, Resize, QLinearMatMul, and Softmax. Profile
instrumentation raises wall time, so it is used for attribution only; the stable
461.603 ms benchmark is the timing authority. Perf/call-graph evidence is
unavailable because `perf` is not installed on the board.
