# Compiler tuning report

All T0-T6 accepted arms built and remained exact. T6 (`-mtune=spacemit-x60 -funroll-loops` while preserving explicit `-march=rv64gcv_zvfh`) was the fastest scout and was selected. `-mcpu=spacemit-x60` parsed and ran, but was not selected because it implicitly changes the ISA contract. LTO, Zvl256b, and max-LMUL arms did not produce a decisive integrated win.

The inline `smt.vmadot` body is opaque to GCC scheduling; no outer-loop gain is attributed to scheduling inside that assembly block.
