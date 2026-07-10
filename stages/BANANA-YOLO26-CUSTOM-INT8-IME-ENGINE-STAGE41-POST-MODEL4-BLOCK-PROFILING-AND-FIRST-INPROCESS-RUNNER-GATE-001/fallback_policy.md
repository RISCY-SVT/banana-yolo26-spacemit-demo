# Fallback Policy

Stage41 fallback is ONNX Runtime CPU through in-process C API sessions.

Allowed role:

```text
correctness oracle
unimplemented block fallback in scaffold
suffix profiling support
```

Forbidden interpretation:

```text
not custom acceleration
not production backend
not model FPS
not final runtime dependency
```

Board fallback currently has a CPU ORT contract mismatch against the accepted Stage40 host ORT oracle. Stage42 must repair or explicitly scope this before accepting new custom block expansion.
