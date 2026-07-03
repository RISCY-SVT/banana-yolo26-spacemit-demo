# Sliding vmadot Ops Note

Stage 1 did not implement `vmadot1`, `vmadot2`, `vmadot3`, `vmadotn`, unsigned variants, or FP/vfmadot variants.

## Evidence summary

Sources read:

- `/control/specs/docs/spacemit-ime-asciidoc.md`
- `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/`
- `/data/lab/task-runs/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001/artifacts/parser/parser-matrix.tsv`
- `/data/lab/task-runs/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001/artifacts/board/board-exec-matrix.tsv`

## Classification

| op | spec-visible | assembler-visible | board-executable | correctness-oracle | implementation-authorized |
|---|---:|---:|---:|---:|---:|
| `vmadot1` | yes | mixed, accepted on some GCC/AS rows | not accepted as Stage 1 evidence | no | no |
| `vmadot2` | yes | mixed, accepted on some GCC/AS rows | not accepted as Stage 1 evidence | no | no |
| `vmadot3` | yes | yes, accepted on usable routes | yes, cluster0 execution proof exists | no independent oracle claim accepted | no |
| `vmadotn` | yes | rejected on tested routes | no | no | no |

## Stage 1 conclusion

Only plain `smt.vmadot` is implementation-authorized for Stage 1. Sliding variants remain future direct-convolution candidates only after separate parser, disassembly, board, and independent oracle proof.
