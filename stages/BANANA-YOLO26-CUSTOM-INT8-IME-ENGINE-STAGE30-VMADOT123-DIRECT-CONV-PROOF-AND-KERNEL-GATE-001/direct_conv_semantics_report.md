# Direct Conv Semantics Report

Stage30 proved instruction-level semantics for `smt.vmadot1/2/3`, but did not accept a real direct Conv kernel.

Reason:

- The proven semantics are shifted 4x4x8 tile semantics over an expanded 8x8 A window.
- A real direct/sliding 3x3 Conv candidate must define how expanded A panels are built, how shifted C tiles are stored without duplicate rows, and how thread partitioning avoids overlap.
- Implementing that correctly is larger than the instruction proof and should be a separate Stage31 kernel-applicability stage.

No model math change was made in Stage30.
