# Sliding Vmadot Ops Note

## Stage 3 Implementation Lane

Stage 3 uses only plain `smt.vmadot` MMT4D:

- `4x4x8`
- signed `int8 x int8 -> int32`
- cluster0 CPU0-3 only

No `vmadot1`, `vmadot2`, `vmadot3`, `vmadotn`, FP, or `vfmadot` instruction was used in Stage 3 code.

## Evidence Paths Searched

- `/control/specs/docs/spacemit-ime-asciidoc.md`
- `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/`
- `/exchange/results/archive/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001/`
- `/data/lab/task-runs/W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001/`

## Status

| op | Stage 3 status |
|---|---|
| `vmadot1` | spec/artifact-visible; future direct-conv/sliding-window candidate only |
| `vmadot2` | spec/artifact-visible; future direct-conv/sliding-window candidate only |
| `vmadot3` | board-executable evidence exists, but no independent oracle authorization for this implementation lane |
| `vmadotn` | rejected/not authorized |
| `vfmadot` / FP variants | blocked/deferred; not part of INT8 custom engine lane |

Plain `smt.vmadot` MMT4D remains the implementation foundation until a separate direct-conv lane is explicitly authorized.
