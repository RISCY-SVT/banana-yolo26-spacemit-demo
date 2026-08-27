# I/O Binding capability

Status: `unsupported`.

The result is based on the shipped target perf-test API, not generic ORT documentation.

Only input pre-binding was exercised. Output device pre-allocation and a device-resident handoff into the separate CPU FP32 tail were not established; the accepted tail requires six host-readable float tensors.
