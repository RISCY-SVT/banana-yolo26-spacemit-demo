# FRM Rounding Regression Report

The accepted A3 candidate was tested on board with ambient RISC-V `frm` values:

```text
RNE: mismatches=0 max_abs_diff=0 after_frm=0 checksum=106597930
RTZ: mismatches=0 max_abs_diff=0 after_frm=1 checksum=106597930
RDN: mismatches=0 max_abs_diff=0 after_frm=2 checksum=106597930
RUP: mismatches=0 max_abs_diff=0 after_frm=3 checksum=106597930
RMM: mismatches=0 max_abs_diff=0 after_frm=4 checksum=106597930
```

The runner path preserves post-call `frm` and does not rely on ambient rounding mode for accepted output bytes.
