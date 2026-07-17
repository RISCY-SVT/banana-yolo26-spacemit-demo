# Hot Disassembly Difference

Symbol-specific dumps and section images are retained under the Stage59 raw
log root in `elf-compare/`. The accepted-flags and published Stage58 binaries
do not have the same hot `.text`:

- published `.text`: 241,652 bytes,
  SHA-256 `e920ef38343e6c9517ceaea9bb9ce08ddc42137004519704f9bf3e9db2e1dc3b`;
- accepted-flags `.text`: 320,558 bytes,
  SHA-256 `d17a5ed182d6905644b192f89c071dcd2b10cae55eb7d67a60edc4a534666c97`.

Hot symbol addresses, loop bodies, and unrolled schedules differ throughout.
This is the expected code-generation consequence of the missing tuning/unroll
flags. No model, package, rounding rule, IME symbol, or executor arithmetic was
changed to repair it.
