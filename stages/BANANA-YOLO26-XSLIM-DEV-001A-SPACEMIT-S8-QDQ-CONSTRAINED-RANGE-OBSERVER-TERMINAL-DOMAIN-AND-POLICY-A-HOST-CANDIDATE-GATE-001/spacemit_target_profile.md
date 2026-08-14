# SpacemiT structural target profile

Profile: `spacemit_k1x_s8_qdq_split_v1`.

The fail-closed validator requires signed INT8 Q/DQ, no QLinear operators, no
UINT8 zero points, per-tensor activation qparams, symmetric per-channel Conv
weights, explicit valid `kernel_shape` for every Conv, exact six output
names/shapes/order, and an exact unquantized-tail hash when supplied.

A pass establishes only graph-format conformance. It does not establish
SpacemiT EP placement, fusion, kernel selection, correctness on K1X,
performance, stability, or production suitability.
