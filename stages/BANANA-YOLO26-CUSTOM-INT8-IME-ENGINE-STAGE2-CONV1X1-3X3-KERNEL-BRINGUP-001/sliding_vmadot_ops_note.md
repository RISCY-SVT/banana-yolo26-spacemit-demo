# Sliding Vmadot Ops Note

Plain `smt.vmadot` MMT4D remains the Stage 2 implementation foundation.

Evidence read:

- `/control/specs/docs/spacemit-ime-asciidoc.md`
- Stage 0 reports under `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE0-RECOVERY-AND-DESIGN-001/`
- Stage 1 reports under `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE1-SMTVMADOT-MICROKERNEL-001/`

Local spec lines show `vmadot1`, `vmadot2`, `vmadot3`, and `vmadotn` mnemonics as architecture-visible sliding variants. Stage 1 accepted only plain `smt.vmadot` as implementation-authorized.

| op | spec-visible | assembler-visible | board-executable | correctness-oracle | implementation-authorized in Stage 2 |
| --- | --- | --- | --- | --- | --- |
| `vmadot1` | yes | not re-tested | not re-tested | no accepted oracle | no |
| `vmadot2` | yes | not re-tested | not re-tested | no accepted oracle | no |
| `vmadot3` | yes | evidence exists for board execution in older artifacts | not re-tested | no independent accepted oracle | no |
| `vmadotn` | yes | rejected/not authorized on tested routes | no accepted route | no | no |
| FP/`vfmadot*` | yes | blocked/deferred | not used | no | no |

Conclusion: `vmadot1/2/3` remain future direct-conv/sliding-window candidates only. Stage 2 did not implement or benchmark them.
