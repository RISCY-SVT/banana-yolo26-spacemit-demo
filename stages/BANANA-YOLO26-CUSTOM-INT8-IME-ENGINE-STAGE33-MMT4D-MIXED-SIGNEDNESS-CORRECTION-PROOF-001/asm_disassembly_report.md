# ASM Disassembly Report

Named asm route was used. No raw opcode route was used.

Disassembly excerpt from `liby26_k1x_custom_int8_engine.a`:

```text
15961:  5c: e2101e2b  smt.vmadotus v28,v0,v1
15972:  7e: e2101e2b  smt.vmadotus v28,v0,v1
```

Baseline and older proof instructions are also visible:

```text
smt.vmadot
smt.vmadot1
smt.vmadot2
smt.vmadot3
```

Raw log:

```text
/data/ncnn-logs/ai-team/2026-07-08_14-41-34/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE33-MMT4D-MIXED-SIGNEDNESS-CORRECTION-PROOF-001/run_logs/asm_disassembly_vmadotus.log
```
