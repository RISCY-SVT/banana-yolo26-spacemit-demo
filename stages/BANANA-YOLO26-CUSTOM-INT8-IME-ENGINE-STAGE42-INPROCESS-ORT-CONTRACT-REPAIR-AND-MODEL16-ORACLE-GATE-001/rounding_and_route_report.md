# Rounding And Route Report

## Custom route

The selected custom arm is the existing Stage39 model4 route:

```text
mode: ime_threaded
merge/dataflow: Y26_STAGE16_MERGE_MODE_STAGE39_BRANCH3X3_FAST_PACK
activation: Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT
output quantize: Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE
threading: branch0=4, branch1=4, model4_cv2=4
affinity: CPU0-3 only
```

The final cross runner disassembly contains the expected `smt.vmadot` instructions, including the 4-accumulator and 6-accumulator helpers. Relevant excerpts are in `model4_route_disassembly.txt`; the complete objdump SHA-256 is `88a295693de26e93bf62e33ec9c1a6a929bd3fdca45cf8d5f39b52a5e5f2d7c9` in raw artifacts.

No CPU4-7 IME work was launched. `taskset -c 0-3` was used and the engine worker affinity check returned `affinity_ok=1`. No SIGILL occurred.

## Rounding restoration

Stage42 same-input scalar and IME calls recorded C `fegetround()` as `0` before and after. The explicit board FRM sweep ran ambient RNE/RTZ/RDN/RUP/RMM (`0/1/2/3/4`); every arm had `mismatches=0`, `max_abs_diff=0`, and `after_frm` equal to the ambient input value. Output checksum was stable at `106597930` for the NHWC fixture.

This confirms the already implemented route and restoration behavior. It is not a new ISA proof and does not authorize new vmadot variants.
