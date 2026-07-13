# Explicit vector and assembly policy

Portable scalar C++ remains the arithmetic authority. Selected performance paths must use
explicit standard-RVV intrinsics/assembly or the already proven explicit SpacemiT IME assembly,
and must have disassembly, exact-oracle, board-execution, and stable-timing evidence. Compiler
auto-vectorization reports are diagnostics only. CPU4-7 binaries must contain and execute no IME.
