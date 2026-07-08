# Duplicate Output Row Policy Report

The Stage31 direct sidecar uses an output M-step of 7.

For each 8-row A-panel and 4-output-channel B tile:

- `smt.vmadot` produces rows 0, 1, 2, 3.
- `smt.vmadot1` produces shifted rows where only row 3 is stored as output row 4.
- `smt.vmadot2` produces shifted rows where only row 3 is stored as output row 5.
- `smt.vmadot3` produces shifted rows where only row 3 is stored as output row 6.

Rows produced but not stored are duplicate/overlap work. This is correctness-safe but not performance-efficient.

Measured result:

- Correctness: exact.
- Speed: failed.
- Direct kernel compute is `15795.9 us`, but the full direct node is `56980.9 us`.

Conclusion:

The duplicate-row policy is acceptable as a proof of real-node semantics, but it is not an accepted performance path.
