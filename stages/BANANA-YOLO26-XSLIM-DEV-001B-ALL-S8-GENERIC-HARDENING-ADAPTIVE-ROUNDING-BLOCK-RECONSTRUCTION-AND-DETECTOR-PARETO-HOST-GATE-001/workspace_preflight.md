# XSLIM-DEV-001B workspace preflight

- Gate 0: `pass`.
- Banana effective start: `16d36c569e267016cecabc6333515d2feecb12aa` (accepted-prior-append-only-erratum).
- XSlim start/tree: `3e275c6496d603d3f75f363ed00aa633ffc00408` / `acdd6d64f35c7554f2559c781c5cbe0806acac1a`.
- Stage65C-R1 packet: `8398831b147cc890436e968d830b14c0d5347ee5a24946b03156c66aa08b22e6`, 63 files, 1983169 bytes.
- B2, A1, range manifest, common tail, dataset lists/annotations and protected states match.
- C200 was deterministically split into 160 optimization and 40 reconstruction-validation images; overlap with H500/val2017 is zero.
- Commands use container `/data`; `/exchange` is a separate managed handoff surface.
