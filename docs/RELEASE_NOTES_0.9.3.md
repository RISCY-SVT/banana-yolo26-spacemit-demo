# Release Notes 0.9.3

0.9.3 is a narrow scheduler-liveness maintenance release for the frozen R640 `K1X_INT8_V1` executor.

It repairs frame-gated worker park/wake acknowledgement and threaded-convolution startup readiness publication. ABI1, SOVERSION1, arithmetic, model/package/profile identities, output semantics, camera behavior, and operator selection are unchanged.

The Stage60 resolution sweep is not included. Runtime delivery still excludes source ONNX, and external source-model redistribution remains not cleared.
