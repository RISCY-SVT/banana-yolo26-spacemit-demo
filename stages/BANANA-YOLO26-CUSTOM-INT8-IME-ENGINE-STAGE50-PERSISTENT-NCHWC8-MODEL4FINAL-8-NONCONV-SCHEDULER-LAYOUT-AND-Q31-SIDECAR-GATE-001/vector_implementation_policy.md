# Explicit vector implementation policy

Every selected hot path retains an independent package oracle and portable scalar path. Performance selection requires explicit RVV or constrained assembly in source, confirming disassembly, exact bytes, and stable wall-time improvement. Compiler vectorization diagnostics are not execution proof.

The selected Conv keeps explicit `smt.vmadot` assembly and P3 segmented/strided A delivery. The selected LUT uses explicit indexed RVV loads (`vluxei8.v`). Q62 and Q31 limb-product sidecars use explicit widening RVV multiplies, but neither implements the complete exact vector round/LUT pipeline and neither is selected.
