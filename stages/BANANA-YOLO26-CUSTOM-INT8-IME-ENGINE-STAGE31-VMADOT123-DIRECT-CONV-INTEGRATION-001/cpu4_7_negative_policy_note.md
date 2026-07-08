# CPU4-7 Negative Policy Note

Stage31 did not execute IME code on CPU4-7.

Reason:

- The task explicitly forbids CPU4-7 IME execution.
- Stage31 evidence is limited to cluster0 CPU0-3.

Future negative testing:

Any CPU4/5 negative/protection test must be explicitly authorized and must use a SIGILL-safe probe that cannot be confused with production-like IME execution.
