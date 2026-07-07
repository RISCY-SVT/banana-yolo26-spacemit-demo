# vmadot1/2/3 Future Lane Decision

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE28-MODEL4-CONV-MMT4D-TILE-PREPACK-REPAIR-001`

## Stage28 Evidence

Stage28 T2 removed the local corrected-buffer copy/writeback bucket, but the selected `/model.4` cut remains Conv/MMT4D-compute dominated:

```text
total_us_after_t2: 40231.6
conv_us_after_t2: 25255.4
conv_share_after_t2: 62.775%
conv_compute_us_after_t2: 18097.1
aggregate_conv_gmac_s_after_t2: 5.44935
```

This is structural low utilization for the current plain `smt.vmadot` MMT4D path.

## Decision

`vmadot1/2/3` is not implemented in Stage28.

A future proof lane may be justified only if both gates are true:

1. Stage28 structural evidence remains accepted after review.
2. Track B YOLO26 vendor-ORT rt204 mAP/value baseline shows YOLO26 is worth deeper custom-engine kernel investment.

## Future Proof Lane Requirements

If authorized later, the future lane should be:

```text
BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-VMADOT123-SEMANTICS-AND-CONV-APPLICABILITY-001
```

Required evidence:

```text
spec/source review
assembler/parser acceptance
objdump/disassembly proof
board CPU0-3 execution
CPU4/5 negative policy if safe
exact scalar oracle
comparison vs current threaded MMT4D on one real dominant Conv node
no graph integration until semantics and speed are proven
```

`vmadotn` remains not authorized.
