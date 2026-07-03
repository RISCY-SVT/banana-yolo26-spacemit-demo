# ASM Parser And Disassembly Report

classification: named-mnemonic

## Build route

- compiler: `/data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-g++`
- assembler: `/data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-as`
- objdump: `/data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-objdump`
- sysroot: `/data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/sysroot`
- flags: `-march=rv64gcv_zvfh -mabi=lp64d`
- CMake option: `-DY26_K1X_ENABLE_IME=ON`

## ASM route

- named `smt.vmadot`: accepted
- raw `.insn`: not used
- raw opcode fallback: not used

## Disassembly evidence

Command:

```bash
riscv64-unknown-linux-gnu-objdump -d .deps/custom_int8_engine/build-k1x-vmadot/tests/test_vmadot_4x4x8_board_probe | rg -n -A8 -B8 "vmadot|smt\\.vmadot"
```

Excerpt from `test_vmadot_4x4x8_board_probe`:

```text
10e7c: 02050007           vle8.v      v0,(a0)
10e80: 02058087           vle8.v      v1,(a1)
10e84: e2103e2b           smt.vmadot  v28,v0,v1
10e88: 011072d7           vsetvli     t0,zero,e32,m2,tu,mu
10e8c: 02066e27           vse32.v     v28,(a2)
...
10ea2: 02058087           vle8.v      v1,(a1)
10ea6: e2103e2b           smt.vmadot  v28,v0,v1
10eaa: 011072d7           vsetvli     t0,zero,e32,m2,tu,mu
10eae: 02066e27           vse32.v     v28,(a2)
```

The same named instruction is present in `bench_vmadot_microkernel`.
