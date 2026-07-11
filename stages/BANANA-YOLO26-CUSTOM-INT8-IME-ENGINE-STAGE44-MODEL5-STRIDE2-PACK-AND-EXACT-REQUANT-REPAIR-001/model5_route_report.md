# Model5 route

R2a is selected only by `Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK`. It calls the existing stage39 fast-pack MMT4D entry with a new guarded 3x3s2 chunk pack; R0 calls the Stage37 pipeline. Both preserve signed storage, `smt.vmadot` s8xs8 accumulation, explicit correction, fixed integer requant, and SiLU LUT.

Guard: kernel 3x3, stride 2, pad 1, input channels divisible by 8, exact K padding, four valid output positions on one row. Border tiles fill the signed storage zero point. Tail/unsupported tiles fall back to the existing generic pack.

The RISC-V binary was built with `-march=rv64gcv_zvfh -mabi=lp64d`, SpacemiT toolchain under `/opt/riscv`, and IME enabled. `model5_route_disassembly.txt` records the fast-pack body and existing `smt.vmadot` words. Execution was CPU0-3 only and produced no SIGILL.
