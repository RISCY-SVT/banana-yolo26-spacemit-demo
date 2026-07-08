# vmadot123 Disassembly Report

Objdump command:

`/data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-objdump -d .deps/custom_int8_engine/build-riscv-stage30/bench_stage30_vmadot123_probe | grep -E "smt\.vmadot|\.insn|\.word"`

Observed symbolic instructions:

```text
 ### command: /data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-objdump -d .deps/custom_int8_engine/build-riscv-stage30/bench_stage30_vmadot123_probe | grep -E "smt\.vmadot|\.insn|\.word"
    16afa:	e7043e2b          	smt.vmadot1	v28,v8,v16
    16b26:	e7047e2b          	smt.vmadot2	v28,v8,v16
    16b44:	e704be2b          	smt.vmadot3	v28,v8,v16
    16bf4:	e2103e2b          	smt.vmadot	v28,v0,v1
    16c16:	e2103e2b          	smt.vmadot	v28,v0,v1
```

Result:

- `smt.vmadot1 v28,v8,v16`: disassembly-visible.
- `smt.vmadot2 v28,v8,v16`: disassembly-visible.
- `smt.vmadot3 v28,v8,v16`: disassembly-visible.
- No raw `.insn` or `.word` substitution was needed.
