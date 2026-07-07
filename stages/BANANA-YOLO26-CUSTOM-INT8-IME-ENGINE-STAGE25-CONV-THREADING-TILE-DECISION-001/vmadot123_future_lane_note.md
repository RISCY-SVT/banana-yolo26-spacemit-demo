# vmadot123 Future Lane Note

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE25-CONV-THREADING-TILE-DECISION-001

## Policy

```text
vmadot1/2/3 implementation: not authorized in Stage25
vmadotn: not authorized
FP/vfmadot: not authorized
accepted Stage25 implementation foundation: plain integer smt.vmadot MMT4D
```

## Decision

Stage25 does not open a vmadot1/2/3 proof lane as the immediate next stage because C1 cluster0 threading reduced the selected Conv bucket and Conv is no longer dominant.

Future vmadot1/2/3 proof should only be opened if:

```text
- representative/full-shape Conv becomes dominant again after activation/merge repairs;
- plain MMT4D remains structurally low-utilization on a real dominant Conv;
- a separate proof stage provides parser/assembler/disassembly, CPU0-3 board execution, scalar oracle, and per-node comparison;
- the future proof clears a >=2x per-node speedup threshold against the then-current selected path.
```

This note makes no model FPS or production readiness claim.
