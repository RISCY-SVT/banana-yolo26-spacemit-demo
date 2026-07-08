# vmadotn Recheck Report

Scope:

`vmadotn` was rechecked as a proof-only sidecar. It was not integrated into the engine and was not used by the Stage31 direct Conv candidate.

String surface:

- `smt.vmadotn`: not visible in assembler/objdump strings.
- `smt.vmadot.n`: not visible.
- `smt.vmadot4`: not visible.

Assembler matrix:

| Candidate mnemonic | Status | Evidence |
| --- | --- | --- |
| `smt.vmadotn` | rejected-with-stderr | `unrecognized opcode` |
| `smt.vmadot.n` | rejected-with-stderr | `unrecognized opcode` |
| `smt.vmadot4` | rejected-with-stderr | `unrecognized opcode` |

Board execution:

Not attempted, because no parser/assembler/disassembly-safe route exists for these candidate mnemonics.

Classification:

`rejected-with-stderr`

Conclusion:

`vmadotn` remains not authorized and not usable for engine work in this branch.
