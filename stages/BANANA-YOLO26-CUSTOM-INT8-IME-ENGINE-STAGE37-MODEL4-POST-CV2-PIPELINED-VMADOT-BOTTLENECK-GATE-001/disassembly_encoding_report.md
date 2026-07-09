# Disassembly Encoding Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001

## Scope

Stage37 introduced a new wrapper path that reuses the Stage36 software-pipelined `smt.vmadot` MMT4D core for branch 3x3 Conv nodes. No new raw opcode route was introduced.

## Evidence

The RISC-V binary was disassembled after the cross build. The objdump output contains:

```text
y26_threaded_conv_run_ime_cluster0_stage37_pipelined
conv_stage37_pipelined_core(...)
y26_conv2d_i8s8s32_nhwc_ime_prepacked_stage37_pipelined_v1
run_c_tiles_stage36_pipelined4
smt.vmadot
```

Full raw evidence:

```text
/data/ncnn-logs/ai-team/2026-07-09_07-02-24/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE37-MODEL4-POST-CV2-PIPELINED-VMADOT-BOTTLENECK-GATE-001/run_logs/objdump_stage37_vmadot.log
```

## Register / Instruction Policy

```text
instruction: named smt.vmadot through the existing Stage36 accepted helper path
operand family: signed s8 x signed s8 -> s32
vmadotus: not selected
vmadot1/2/3 direct/sliding: not selected
vmadotn: not used
```

## Result

```text
disassembly_status: pass
board_execution_status: pass
sigill: none observed in Stage37 candidate smoke or stable run
```
