# Rounding Mode Regression Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE22-MODEL4-ONNX-CUT-ORACLE-AND-NEXT-BLOCK-GATE-001`

## Finding

The first board sweep exposed one mismatch when ambient `frm=RTZ`. Stage22 repaired this by adding a scoped RNE guard around the selected Stage22 same-input runner invocation and restoring the caller's ambient `frm` after the invocation. This keeps the float-domain merge and quantization behavior stable against ONNX Runtime CPU reference semantics for this verifier path.

## Final Board Sweep

```text
ambient_frm=0 RNE status=0 mismatches=0 max_abs_diff=0 after_frm=0 checksum=106597930
ambient_frm=1 RTZ status=0 mismatches=0 max_abs_diff=0 after_frm=1 checksum=106597930
ambient_frm=2 RDN status=0 mismatches=0 max_abs_diff=0 after_frm=2 checksum=106597930
ambient_frm=3 RUP status=0 mismatches=0 max_abs_diff=0 after_frm=3 checksum=106597930
ambient_frm=4 RMM status=0 mismatches=0 max_abs_diff=0 after_frm=4 checksum=106597930
```

Conclusion:

```text
rounding_regression_status: pass
post_call_frm_unchanged: true
```
