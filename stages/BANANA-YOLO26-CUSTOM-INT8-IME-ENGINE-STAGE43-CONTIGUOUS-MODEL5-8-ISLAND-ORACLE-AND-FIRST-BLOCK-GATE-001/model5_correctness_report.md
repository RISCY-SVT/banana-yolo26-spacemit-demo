# Model5 Correctness Report

The model4 final activation and model5 output were checked at exact integer boundaries for F0-F7.

- host scalar vs semantic fixed-host cut: 0 mismatches for all fixtures;
- board scalar vs host scalar: 0 mismatches for all fixtures;
- board IME vs board scalar: 0 mismatches for all fixtures;
- board IME vs semantic fixed-host cut: 0 mismatches for all fixtures;
- repeated hashes: stable;
- FRM RNE/RTZ/RDN/RUP/RMM: all pass with ambient `frm` restored;
- worker affinity: CPU0-3, `affinity_ok=1`;
- SIGILL: none.

The initial RVV-float model5 requant differed at one F7 element although Conv int32 buffers were identical. Replacing only that handoff with prepared fixed-point requant plus the exact LUT closed the mismatch. No tolerance was accepted.

Operational `ORT_ENABLE_ALL` differs from semantic `ORT_DISABLE_ALL` for stress fixtures because of the separately proven x86 optimized-kernel saturation behavior. That artifact is not treated as a K1X custom-kernel correctness target.
