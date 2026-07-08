# Objdump VMADOT Report

binary: `.deps/custom_int8_engine/build-riscv-stage34/bench_stage34_vmadot_throughput`

raw logs:

```text
run_logs/objdump_stage34_vmadot.out
run_logs/objdump_stage34_vmadot_cases.out
run_logs/objdump_stage34_safe_cases.out
run_logs/objdump_stage34_exact.out
```

Symbolic disassembly contains only named forms for this Stage34 diagnostic:

```text
smt.vmadot v20,v0,v1
smt.vmadot v22,v0,v1
smt.vmadot v24,v0,v1
smt.vmadot v26,v0,v1
smt.vmadot v28,v0,v1
smt.vmadot v30,v0,v1
```

The binary also contains existing Stage33 `smt.vmadotus` wrappers from linked library code, but Stage34 did not select or continue that candidate.

No raw `.insn` / `.word` vmadot implementation was introduced.
