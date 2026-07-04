# Sliding VMADOT Future Lane Note

Stage 11 implementation primitive remains:

`smt.vmadot 4x4x8 s8xs8->s32` through the MMT4D Conv path.

## Not Implemented In Stage 11

- `vmadot1`
- `vmadot2`
- `vmadot3`
- `vmadotn`
- FP/vfmadot

## Policy

`vmadot1/2/3` may be future direct-convolution or sliding-window candidates, but any use requires a separate stage with artifact recovery, assembler/parser evidence, disassembly, board execution, independent oracle proof, and acceptance gates.

`vmadotn` remains not authorized based on prior tested routes.

No sliding op was silently substituted into current Conv kernels.
