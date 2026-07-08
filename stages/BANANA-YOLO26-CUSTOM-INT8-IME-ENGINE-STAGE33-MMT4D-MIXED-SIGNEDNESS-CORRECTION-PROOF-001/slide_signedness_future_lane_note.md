# Slide Signedness Future Lane Note

Stage30/31 proved base `smt.vmadot1/2/3` shifted-row semantics for signed operands.

Stage32 proved non-slide integer dot family mnemonics:

```text
smt.vmadot:   s8 x s8
smt.vmadotu:  u8 x u8
smt.vmadotsu: s8 x u8
smt.vmadotus: u8 x s8
```

Stage33 used only non-slide `smt.vmadotus`.

`vmadot1/2/3` signedness variants remain report-only unless independently proven. A future proof must handle:

```text
vd/vs1 even constraints
non-overlap constraints
implicit t0 slide value and clobber policy
board CPU0-3 execution
independent scalar oracle
real-node comparison against current MMT4D
```

No slide-family signedness variant is implemented or selected in Stage33.
