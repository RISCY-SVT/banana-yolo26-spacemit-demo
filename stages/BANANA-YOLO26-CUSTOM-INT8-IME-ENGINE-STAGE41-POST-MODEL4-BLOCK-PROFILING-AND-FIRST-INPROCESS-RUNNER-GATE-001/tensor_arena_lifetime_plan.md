# Tensor Arena Lifetime Plan

Target arena policy:

```text
allocate_once: true
hot_loop_heap_allocation: false
tensor_residency: quantized tensors remain in memory across blocks
temporary_lifetime: per-block scratch slices reused by static schedule
alignment: 64 bytes for IME/packed layouts unless a kernel requires stricter alignment
```

Initial arena classes:

```text
input/output resident tensors
custom block internal activations
prepacked weight buffers
thread-local workspaces for CPU0-3 IME workers
ORT fallback bridge tensors while fallback remains in scaffold
```

Stage42 should avoid adding new file-backed tensor handoff.
