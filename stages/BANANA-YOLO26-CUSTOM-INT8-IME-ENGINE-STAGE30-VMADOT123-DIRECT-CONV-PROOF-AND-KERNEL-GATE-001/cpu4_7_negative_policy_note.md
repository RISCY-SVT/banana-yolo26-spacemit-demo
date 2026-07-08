# CPU4-7 Negative Policy Note

Stage30 did not intentionally execute IME instructions on CPU4-7.

Rationale:

- Prior custom-engine policy is cluster0-only for IME: CPU0, CPU1, CPU2, CPU3.
- The Stage30 prompt did not require a CPU4-7 negative probe.
- A negative probe would need a SIGILL-safe, explicitly authorized route and is not necessary for `vmadot1/2/3` semantics proof on the allowed CPUs.

The proof-only helper uses the existing cluster0 runtime guard for checked execution. No production-like IME path is run on CPU4-7.
