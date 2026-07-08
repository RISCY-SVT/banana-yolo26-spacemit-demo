# vmadot123 Parser / Assembler Report

Stage ID: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001`

Toolchain route:

`/data/SpacemiT/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-g++`

CMake route:

`-DY26_K1X_ENABLE_IME=ON -DCMAKE_CXX_FLAGS=-march=rv64gcv_zvfh -mabi=lp64d`

Result:

| variant | named asm | assembler result | notes |
| --- | --- | --- | --- |
| `smt.vmadot1` | yes | pass | emitted as symbolic instruction |
| `smt.vmadot2` | yes | pass | emitted as symbolic instruction |
| `smt.vmadot3` | yes | pass | emitted as symbolic instruction |
| `smt.vmadotn` | not attempted | not authorized | remains rejected/not authorized |

A first cross-build attempt without explicit RISC-V vector arch flags failed because the assembler rejected existing IME/vector instructions. That route is recorded as invalid. The accepted route is the explicit `rv64gcv_zvfh/lp64d` route above.

Raw logs:

- `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001/run_logs/riscv_configure_stage30_after_oracle_patch.log`
- `/data/ncnn-logs/ai-team/2026-07-08_09-43-20/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE30-VMADOT123-DIRECT-CONV-PROOF-AND-KERNEL-GATE-001/run_logs/riscv_build_stage30_after_a_window_fix.log`
