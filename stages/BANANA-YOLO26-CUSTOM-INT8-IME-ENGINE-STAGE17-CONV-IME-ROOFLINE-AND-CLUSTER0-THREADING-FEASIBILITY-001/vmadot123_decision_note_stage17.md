# vmadot1/2/3 Decision Note Stage17

Current accepted state:

```text
smt.vmadot: accepted implementation primitive
vmadot1/2/3: possible future direct-conv/sliding-window lane only
vmadotn: rejected/not authorized
FP/vfmadot: not part of this INT8 lane
```

Stage17 did not implement `vmadot1`, `vmadot2`, `vmadot3`, or `vmadotn`.

Reason: Stage17 first had to repair benchmark methodology and test cluster0 threading. The result is `strong_positive` for 4-thread spatial row split, so the next immediate stage should integrate bounded cluster0 threading before opening a sliding-op proof lane.

Future trigger:

```text
If Conv/IME remains dominant after stable threaded integration,
or if threading cannot carry later representative/full-shape nodes,
open:
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-VMADOT123-SEMANTICS-AND-CONV-APPLICABILITY-001
```

Required future proof:

```text
parser/assembler/disassembly
board CPU0-3 execution
controlled CPU4/5 negative if safe
scalar oracle
comparison against current MMT4D path on one real Conv node
>=2x per-node speedup threshold to proceed
```
