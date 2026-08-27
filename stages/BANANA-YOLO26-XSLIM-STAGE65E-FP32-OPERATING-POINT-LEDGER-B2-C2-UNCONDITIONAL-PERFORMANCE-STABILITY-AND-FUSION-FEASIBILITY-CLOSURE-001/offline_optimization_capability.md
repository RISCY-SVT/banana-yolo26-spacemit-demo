# Offline optimization capability

Status: `unsupported`.

The shipped target `onnxruntime_perf_test` was invoked on the physical K1X with the exact SpaceMIT provider. It did not complete valid create-and-readback for both frozen models.

Artifacts remain raw diagnostic bytes and do not replace B2 or C2. This capability alone does not prove a startup or steady-state benefit.
