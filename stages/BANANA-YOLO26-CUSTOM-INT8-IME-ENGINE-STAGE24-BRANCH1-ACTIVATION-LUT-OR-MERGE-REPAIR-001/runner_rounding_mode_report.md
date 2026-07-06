# runner_rounding_mode_report

Runner path:

```text
y26_stage16_model4_c2f_run_cut_u8_output
merge_repair: split1_lut
output_quantize: rvv
```

Board ambient `frm` sweep:

```text
RNE: status=0 mismatches=0 max_abs_diff=0 after_frm=0 checksum=106597930
RTZ: status=0 mismatches=0 max_abs_diff=0 after_frm=1 checksum=106597930
RDN: status=0 mismatches=0 max_abs_diff=0 after_frm=2 checksum=106597930
RUP: status=0 mismatches=0 max_abs_diff=0 after_frm=3 checksum=106597930
RMM: status=0 mismatches=0 max_abs_diff=0 after_frm=4 checksum=106597930
```

Status: `pass`.

The accepted runner path keeps the Stage23 scoped RNE guard and explicit RNE output QuantizeLinear path.
