# Spec Constraints Report

stage_id: `BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE35-VMADOT-SIGILL-EMISSION-REPAIR-AND-THROUGHPUT-001`

## Sources Read

Local source/spec mirror:

- `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/2026-06-27_07-00-01_W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001-TIER2-REVIEW-AND-CLOSURE-001/final-report.md`
- `/control/specs/drafts/0003-ncnn-int8-ime-mmt4d/2026-06-27_07-00-01_W1-IME-INT8-VMADOT-TOOLCHAIN-HARDWARE-PROOF-001-TIER2-REVIEW-AND-CLOSURE-001/summary.md`
- `stages/BANANA-YOLO26-CUSTOM-INT8-IME-ENGINE-STAGE32-MMT4D-OR-LOW-OVERHEAD-SLIDING-DATA-LAYOUT-DECISION-001/integer_dot_signedness_family_report.md`
- `custom_int8_engine/kernels/vmadot_4x4x8_ime.cpp`

Public references checked:

- `https://github.com/spacemit-com/riscv-ime-extension-spec`
- `https://www.remlab.net/op/riscv-xstime.shtml`

The GitHub public spec repository states that the IME proposal reuses vector registers and supports vector register VLEN from 128 to 4096 bits. The Remlab public article describes `vmadot` as a widening integer matrix multiply/accumulate using vector register operands and a 32-bit integer destination.

## Constraints Enforced For Stage35

- Inputs are vector operands `VS1` and `VS2`; the Stage35 helper convention uses A in `v0`/VS1 and B in `v1`/VS2.
- Integer output C is stored in two sequential vector registers because the accumulator matrix is 32-bit and has effective LMUL=2.
- `vd` must be even.
- For X60/VLEN=256/SEW=8, plain `smt.vmadot` is treated as the accepted 4x4x8 int8-to-int32 MAC tile from Stage1/32 evidence.
- The accepted helper uses `vsetvli t0, zero, e32, m2` for accumulator zero/store views and `vsetvli t0, zero, e8, m1` for A/B execution.
- CPU0 first, then CPU0-3 only. No CPU4-7 IME execution.
- No `vmadotn`, no vfmadot/FP IME lane, no int4/int16 lane, no graph/full-engine/default backend work.

## Toolchain Feature Probe

Observed on Stage35 preflight:

```text
-march=rv64gcv_zvfh_xsmtvdot -mabi=lp64d: rejected, unsupported non-standard extension `xsmtvdot`
-mcpu=spacemit-x60 -mabi=lp64d: accepted for a trivial compile probe
```

Therefore Stage35 does not depend on a new `-march=..._xsmtvdot` spelling. It uses the same project route already used by the accepted helper and confirms behavior by objdump and board execution.
