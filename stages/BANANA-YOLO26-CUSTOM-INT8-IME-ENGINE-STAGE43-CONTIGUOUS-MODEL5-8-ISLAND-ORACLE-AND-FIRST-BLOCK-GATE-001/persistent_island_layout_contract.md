# Persistent Island Layout Contract

Logical ONNX tensors remain NCHW uint8. Custom physical storage is NHWC.

The model5 API consumes the current model4 runner's NHWC uint8 preactivation output directly. It performs model4 final activation/requant into persistent NHWC signed-code storage, runs model5 Conv and requant in NHWC, and leaves model5 output in NHWC signed-code storage.

Invariants:

- one NCHW-to-NHWC conversion at the current model4 island entry;
- zero materialized NCHW/NHWC transposes between model4 and model5;
- zero intermediate float tensor materialization;
- at most one NHWC-to-NCHW conversion at the current model5 island exit;
- correctness dumps convert outside measured custom compute;
- scale, zero point, logical name, and physical code convention are explicit.

The contract and direct API handoff pass. A full model4-to-model5 hybrid scaffold was not benchmarked because model5 failed its earlier compute gate.
