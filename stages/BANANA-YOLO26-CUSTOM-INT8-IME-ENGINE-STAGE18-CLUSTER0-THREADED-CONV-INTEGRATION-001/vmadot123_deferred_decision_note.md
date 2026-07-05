# vmadot1/2/3 Deferred Decision Note

Stage18 did not implement `vmadot1`, `vmadot2`, `vmadot3`, or `vmadotn`.

Current lane:

```text
smt.vmadot MMT4D remains the implementation primitive
Stage17 found low single-thread utilization and strong cluster0 threading
Stage18 integrates proven cluster0 threading first
```

Future lane:

```text
vmadot1/2/3 direct-conv/sliding remains a separate proof lane
vmadotn remains not authorized
```

Future proof threshold:

```text
>=2x per-node speedup vs threaded or single-thread MMT4D baseline
parser/assembler/disassembly proof
board CPU0-3 execution proof
controlled CPU4/5 negative if safe
scalar oracle
one real Conv node comparison
```

No sliding-op substitution was made in Stage18.
