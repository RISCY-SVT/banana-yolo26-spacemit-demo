# Integer Dot Signedness Family Report

stage_id: BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001
purpose: proof-only audit for matrix dot signedness family

## Scope

This is not a runtime integration. No mixed signedness opcode is selected for the engine in Stage32.

Audited mnemonics:

```text
smt.vmadot
smt.vmadotu
smt.vmadotsu
smt.vmadotus
```

## Parser and Disassembly Evidence

Raw objdump:

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/stage32_objdump_vmadot_family.log
```

Observed disassembly:

```text
smt.vmadot
smt.vmadotu
smt.vmadotus
smt.vmadotsu
smt.vmadot1
smt.vmadot2
smt.vmadot3
```

## Board Oracle Matrix

Stable run:

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/stage32_layout_signedness_board_retry.log
```

CPU0-3 smoke:

```text
/data/ncnn-logs/ai-team/2026-07-08_12-51-18/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/run_logs/stage32_signedness_cpu0_3_board.log
```

| mnemonic | best scalar hypothesis | board traps | oracle mismatches | status |
|---|---|---:|---:|---|
| smt.vmadot | s8xs8 | 0 | 0 | pass |
| smt.vmadotu | u8xu8 | 0 | 0 | pass |
| smt.vmadotsu | s8xu8 | 0 | 0 | pass |
| smt.vmadotus | u8xs8 | 0 | 0 | pass |

The CPU0-3 smoke repeated the same zero-mismatch result for all four mnemonics.

## Applicability to Current Model4 Cut

Current accepted Conv path uses plain signed `smt.vmadot` over signed storage. For model Q/DQ boundaries where activation is naturally `uint8` and weights are `int8`, `smt.vmadotus` maps to an `u8 x s8` dot hypothesis and may reduce activation conversion or correction work. If a future path chooses signed activations and unsigned weights, `smt.vmadotsu` maps to the opposite operand order.

Current Stage32 selected-cut correction:

```text
aggregate_correction_us: 2497.31
model4_cv2_correction_us: 1758.66
selected_cut_total_us: 40646.8
```

The absolute correction bucket is not the largest selected-cut bucket, but it is locally bounded and now has instruction-family proof. The next stage should prove whether mixed signedness can reduce correction or conversion without changing ONNX-cut bytes.

## Future Proof Requirements

Before integration, a future stage must prove:

```text
- exact operand layout and zero-point contract per Conv node;
- whether activation uint8 can be consumed directly without hidden copy/correction;
- scalar oracle equality for full MMT4D tile accumulation;
- same-input ONNX-cut output SHA remains 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433;
- board CPU0-3 correctness;
- stable timing versus current plain smt.vmadot MMT4D.
```
