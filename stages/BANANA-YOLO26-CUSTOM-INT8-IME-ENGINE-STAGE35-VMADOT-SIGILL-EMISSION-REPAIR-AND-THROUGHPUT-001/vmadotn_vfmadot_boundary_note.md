# VMADOTN / VFMADOT Boundary Note

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## vmadotn

```text
status: not authorized
stage35_action: no implementation, no integration
raw_probe: not performed
```

No source-backed opcode encoding was introduced in Stage35. Raw vmadotn probing remains forbidden without explicit source-backed encoding and human approval.

## vfmadot / FP IME

```text
status: not used
stage35_action: no implementation, no integration
```

Stage35 stayed on integer `smt.vmadot` only.

## CPU Policy

```text
allowed IME CPUs: CPU0, CPU1, CPU2, CPU3
forbidden IME CPUs: CPU4, CPU5, CPU6, CPU7
```

Stage35 CPU0-3 smoke remained within cluster0. CPU4-7 were not used for IME execution.
