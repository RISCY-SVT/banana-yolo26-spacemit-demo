# vmadotn Future Applicability Note

Stage31 did not find a parser/assembler/disassembly route for:

- `smt.vmadotn`
- `smt.vmadot.n`
- `smt.vmadot4`

No raw `.insn` route was accepted because this stage requires source-backed instruction evidence before board execution. `vmadotn` remains rejected/not authorized.

Future work may reopen `vmadotn` only if a vendor spec or accepted assembler route identifies an exact mnemonic/encoding and an independent scalar oracle can be built before any engine integration.
