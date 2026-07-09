# Disassembly Encoding Report

Build:

- RISC-V build: `.deps/custom_int8_engine/build-riscv-stage36`
- `Y26_K1X_ENABLE_IME=ON`
- toolchain route: `/opt/riscv/bin/riscv64-unknown-linux-gnu-g++`
- ISA flags used for final cross build: `-march=rv64gcv_zvfh -mabi=lp64d`

Stage36 A1 `smt.vmadot` payload:

| instruction | word |
| --- | --- |
| `smt.vmadot v20,v0,v1` | `e2103a2b` |
| `smt.vmadot v22,v0,v2` | `e2203b2b` |
| `smt.vmadot v24,v0,v3` | `e2303c2b` |
| `smt.vmadot v26,v0,v4` | `e2403d2b` |

Stage36 A2 `smt.vmadot` payload:

| instruction | word |
| --- | --- |
| `smt.vmadot v16,v0,v1` | `e210382b` |
| `smt.vmadot v18,v0,v2` | `e220392b` |
| `smt.vmadot v20,v0,v3` | `e2303a2b` |
| `smt.vmadot v22,v0,v4` | `e2403b2b` |
| `smt.vmadot v24,v0,v5` | `e2503c2b` |
| `smt.vmadot v26,v0,v6` | `e2603d2b` |

All accumulator destination registers are even and represent EMUL=2 pairs. Destination pairs do not overlap input registers.

Raw evidence:

- `/data/ncnn-logs/ai-team/2026-07-09_05-29-08/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE36-CV2-PIPELINED-VMADOT-CANDIDATE-001/artifacts/objdump_stage36_vmadot_snippets.txt`
