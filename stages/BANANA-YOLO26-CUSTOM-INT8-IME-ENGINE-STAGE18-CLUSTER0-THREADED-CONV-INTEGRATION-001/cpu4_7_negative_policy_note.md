# CPU4-7 Negative Policy Note

Stage18 did not run IME kernels on CPU4, CPU5, CPU6, or CPU7.

Policy:

```text
Allowed IME CPUs: CPU0-3
Forbidden IME CPUs: CPU4-7
Negative probe: not required in Stage18
```

The Stage18 sidecar maps worker `i` to CPU `i` for `i in [0, thread_count)`, with `thread_count <= 4`.

No production-like IME code was executed on CPU4-7.
