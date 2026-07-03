# Cluster0 Safety Plan

Accepted IME policy:

```text
IME kernels only on CPU0-3
no IME on CPU4-7
no all-core dispatch
```

## Runtime Plan

- Provide a cluster0 pinning API before any IME kernel dispatch.
- In debug builds, verify the current CPU before executing IME asm.
- Use a SIGILL guard in probe/test binaries.
- If the CPU, OS, or toolchain route is unsupported, fall back to scalar/RVV or
  return a clear unsupported status.
- Report cluster0-only performance with exact affinity, thread count, warmup,
  run count, repeats, model hash, image hash, and output hash.

## Stage 0 Non-Actions

- No IME asm implemented.
- No board system mutation.
- No package installation.
- No all-core benchmark.
