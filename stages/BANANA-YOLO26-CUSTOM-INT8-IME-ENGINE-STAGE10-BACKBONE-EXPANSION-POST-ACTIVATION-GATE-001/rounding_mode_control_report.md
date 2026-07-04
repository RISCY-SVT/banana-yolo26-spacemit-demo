# Rounding Mode Control Report

## Source Change

Stage 10 changed the accepted A2 RVV path from ambient-FRM conversion:

`__riscv_vfcvt_x_f_v_i32m4(vf, vl)`

to explicit RNE conversion:

`__riscv_vfcvt_x_f_v_i32m4_rm(vf, __RISCV_FRM_RNE, vl)`

## Compiler Evidence

Cross compiler: `/opt/riscv/bin/riscv64-unknown-linux-gnu-g++`, GCC `14.3.0`.

The RISC-V object disassembly includes `fsrmi 0` before `vfcvt.x.f.v` and `fsrm` restore after the vector conversion region. Raw evidence is stored in:

`/data/ncnn-logs/ai-team/2026-07-04_15-15-28/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001/run_logs/objdump_stage10_activation.log`

## Board Regression

`test_stage10_rvv_rounding_control` ran on CPU0, CPU1, CPU2, and CPU3. For both small fixtures and ambient `frm` values RNE/RTZ/RDN/RUP/RMM:

- status: `0`
- mismatches: `0`
- post-call `frm`: unchanged from ambient value

## Decision

`rvv_rne_control: pass` for the Stage 9 current boundary and the Stage 10 Conv2-to-Split boundary.
