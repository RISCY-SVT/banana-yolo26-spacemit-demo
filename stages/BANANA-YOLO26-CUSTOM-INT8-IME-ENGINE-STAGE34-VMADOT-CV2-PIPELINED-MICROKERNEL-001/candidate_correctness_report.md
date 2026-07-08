# Candidate Correctness Report

## A0 Current Baseline

```text
same-input ONNX cut: pass
mismatches: 0
max_abs_diff: 0
output_sha256: 70adcf083e85ea95d5fb42e57ea26f51255944aaa442cde4d919f5ae551b3433
frm_sweep: pass RNE/RTZ/RDN/RUP/RMM
affinity_ok: 1
```

## A1-A5

A1-A5 were not implemented into the runner because the mandatory Step 0 microdiagnostic found that the required direct inline/register-blocked `smt.vmadot` shapes were not board-executable in this stage.

No failing A1-A5 candidate was promoted into the selected runtime path.
