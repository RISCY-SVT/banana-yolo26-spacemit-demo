# Stage30 Hypotheses

H1: `smt.vmadot1`, `smt.vmadot2`, and `smt.vmadot3` may be parser/assembler/disassembly visible in the current SpacemiT toolchain, but each variant requires current board execution proof before any engine use.

H2: Board execution alone is insufficient. A variant is implementation-authorized only after an independent scalar oracle explains its output on deterministic fixtures.

H3: The `vmadot1/2/3` family is expected to be related to shifted/sliding 4x4x8 dot-product tiles, not a full direct Conv replacement by itself.

H4: If semantics are proven but no bounded direct 3x3 Conv sidecar can be accepted in Stage30, the correct outcome is a proof packet and a Stage31 integration prompt, not a broad Conv rewrite.

H5: `vmadotn` remains rejected/not authorized unless a separate future stage provides parser, disassembly, board, and oracle proof. Stage30 does not use it.
