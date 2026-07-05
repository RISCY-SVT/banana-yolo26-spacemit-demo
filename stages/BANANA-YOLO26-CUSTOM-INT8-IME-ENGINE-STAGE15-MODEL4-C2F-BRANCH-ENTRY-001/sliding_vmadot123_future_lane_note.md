# Sliding `vmadot1/2/3` Future Lane Note

Stage 15 implementation uses only plain `smt.vmadot` MMT4D through the existing Stage 1-14 Conv path.

## Not Used

- `vmadot1`
- `vmadot2`
- `vmadot3`
- `vmadotn`
- FP/vfmadot

## Status

`vmadotn`:

- not proven;
- not authorized.

`vmadot1/2/3`:

- possible future direct-conv/sliding-window lane;
- likely relevant for small-channel `3x3` Conv if current MMT4D path remains Conv-dominated on representative/full-shape timing;
- not authorized for Stage 15 implementation.

Any sliding-op implementation requires a separate proof stage with:

- parser/assembler evidence;
- disassembly evidence;
- board CPU0-3 proof;
- CPU4/5 negative/protection check only if safe and authorized;
- scalar oracle;
- direct-conv vs current MMT4D comparison.

Stage 15 did not spend time on sliding experiments.
