# Scale Rounding Property Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE10-BACKBONE-EXPANSION-POST-ACTIVATION-GATE-001`

## Rounding Contract

- C++ scalar reference uses nearest-even through `std::nearbyint` with default RNE in the tested environment.
- Stage 10 RVV A2 path uses explicit `_rm(..., __RISCV_FRM_RNE, ...)` for `vfcvt.x.f.v`.
- Board regression changed ambient `frm` through RNE/RTZ/RDN/RUP/RMM and observed `mismatches=0`; the ambient `frm` value was restored after the A2 call.

## New Boundary Property

- boundary: Conv2 corrected int32 -> uint8 conv code -> SiLU LUT -> `Split_output_1` signed storage.
- 256-code ONNX oracle mismatches: `0`
- max_abs_diff_u8: `0`

## Caveat

This proves the selected Stage 10 boundary only. Future boundaries require their own scale/zero-point/LUT oracle.
