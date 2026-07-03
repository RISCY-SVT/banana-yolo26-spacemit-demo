# Sliding Vmadot Ops Note

Stage 4 implementation remains plain `smt.vmadot` MMT4D only.

Status:

- `smt.vmadot`: implementation primitive for Stage 4, `4x4x8 s8xs8->s32`, cluster0 CPU0-3 only.
- `vmadot1`: future direct-conv/sliding-window candidate only; not implemented.
- `vmadot2`: future direct-conv/sliding-window candidate only; not implemented.
- `vmadot3`: future direct-conv/sliding-window candidate only; not implemented.
- `vmadotn`: rejected/not authorized for this implementation lane.
- FP/vfmadot: blocked/deferred, not part of the INT8 custom engine lane.

Artifact/spec sources remain those recorded in Stage 1 through Stage 3 reports. Stage 4 did not re-test sliding variants.
