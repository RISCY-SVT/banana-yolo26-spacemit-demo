# Runner Rounding Mode Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE23-MODEL4-CUT-CLOSED-BUCKET-ATTRIBUTION-AND-TARGETED-REPAIR-001`

## Issue Rechecked

Stage22 found an ambient `frm` sensitivity in the verifier path. Stage23 moved the scoped RNE guard into the real runner cut API so the accepted runner path, not only a bench wrapper, controls FP-to-int quantization behavior.

## Implementation

```text
file: custom_int8_engine/src/model4_c2f_runner.cpp
api: y26_stage16_model4_c2f_run_cut_u8_output
guard: ScopedRiscvRne
scope: full model4 cut runner invocation
post_call_policy: restore caller frm
```

The final output quantization RVV path also uses explicit RNE conversion:

```text
file: custom_int8_engine/kernels/activation_requant.cpp
intrinsic: __riscv_vfcvt_x_f_v_i32m4_rm(..., __RISCV_FRM_RNE, ...)
```

## Board Sweep

Command class:

```text
taskset -c 0-3 ./bench_stage23_model4_runner_cut \
  --fixture-dir fixtures \
  --mode ime_threaded \
  --output-quantize rvv \
  --warmup 0 --runs 1 --repeats 1 \
  --frm-sweep
```

Result:

```text
ambient_frm=0 RNE: status=0 mismatches=0 max_abs_diff=0 after_frm=0 checksum=106597930
ambient_frm=1 RTZ: status=0 mismatches=0 max_abs_diff=0 after_frm=1 checksum=106597930
ambient_frm=2 RDN: status=0 mismatches=0 max_abs_diff=0 after_frm=2 checksum=106597930
ambient_frm=3 RUP: status=0 mismatches=0 max_abs_diff=0 after_frm=3 checksum=106597930
ambient_frm=4 RMM: status=0 mismatches=0 max_abs_diff=0 after_frm=4 checksum=106597930
```

## Conclusion

`H4_rounding_robustness` passed. The real runner API is robust against ambient `frm` for the selected cut path, and it restores the caller rounding mode after the call.
